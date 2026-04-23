module Main where

import qualified AwkernelWorkloadAcceptance as A
import Data.Char (isDigit)
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import System.IO (hPutStrLn, stderr)

splitOn :: Char -> String -> [String]
splitOn delimiter = go []
  where
    go acc [] = [reverse acc]
    go acc (c:cs)
      | c == delimiter = reverse acc : go [] cs
      | otherwise = go (c : acc) cs

natFromInteger :: Integer -> A.Nat
natFromInteger n
  | n <= 0 = A.O
  | otherwise = A.S (natFromInteger (n - 1))

natToInteger :: A.Nat -> Integer
natToInteger A.O = 0
natToInteger (A.S n) = 1 + natToInteger n

natFromField :: String -> Either String A.Nat
natFromField field
  | not (null field) && all isDigit field = Right (natFromInteger (read field))
  | otherwise = Left ("expected natural number, got: " ++ show field)

optionNatFromField :: String -> Either String (A.Option A.JobId)
optionNatFromField "-" = Right A.None
optionNatFromField field = A.Some <$> natFromField field

boolFromField :: String -> Either String A.Bool
boolFromField "true" = Right A.True
boolFromField "false" = Right A.False
boolFromField field = Left ("expected boolean, got: " ++ show field)

listFromCsv :: String -> Either String (A.List A.JobId)
listFromCsv "" = Right A.Nil
listFromCsv csv = listFromFields (splitOn ',' csv)
  where
    listFromFields [] = Right A.Nil
    listFromFields (x:xs) = do
      headNat <- natFromField x
      tailNats <- listFromFields xs
      pure (A.Cons headNat tailNats)

eventFromFields :: String -> String -> String -> Either String A.OpEvent
eventFromFields "Wakeup" a "-" = A.EvWakeup <$> natFromField a
eventFromFields "RequestResched" a "-" = A.EvRequestResched <$> natFromField a
eventFromFields "HandleResched" a "-" = A.EvHandleResched <$> natFromField a
eventFromFields "Choose" a b = A.EvChoose <$> natFromField a <*> natFromField b
eventFromFields "Dispatch" a b = A.EvDispatch <$> natFromField a <*> natFromField b
eventFromFields "Complete" a "-" = A.EvComplete <$> natFromField a
eventFromFields "Stutter" "-" "-" = Right A.EvStutter
eventFromFields tag _ _ = Left ("unsupported event fields: " ++ show tag)

rowFromFields :: [String] -> Either String A.AwkernelCapturedRow
rowFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField] = do
  cpu <- natFromField cpuField
  event <- eventFromFields eventTag eventA eventB
  current <- optionNatFromField currentField
  runnable <- listFromCsv runnableCsv
  needResched <- boolFromField needReschedField
  dispatch <- optionNatFromField dispatchField
  pure (A.MkAwkernelCapturedRow cpu event current runnable needResched dispatch)
rowFromFields fields =
  Left ("expected 8 TSV columns, got " ++ show (length fields) ++ " from " ++ show fields)

rowsFromLines :: [String] -> Either String (A.List A.AwkernelCapturedRow)
rowsFromLines [] = Right A.Nil
rowsFromLines (line:rest)
  | null line = rowsFromLines rest
  | otherwise = do
      row <- rowFromFields (splitOn '\t' line)
      rows <- rowsFromLines rest
      pure (A.Cons row rows)

insertCandidateSorted :: A.JobId -> A.List A.JobId -> A.List A.JobId
insertCandidateSorted j xs =
  case xs of
    A.Nil -> A.Cons j A.Nil
    A.Cons x xs' ->
      case A.eqb0 j x of
        A.True -> A.Cons x xs'
        A.False ->
          case A.leb j x of
            A.True -> A.Cons j (A.Cons x xs')
            A.False -> A.Cons x (insertCandidateSorted j xs')

addCandidateOnce :: A.JobId -> A.List A.JobId -> A.List A.JobId
addCandidateOnce = insertCandidateSorted

addOptionalCandidate :: A.Option A.JobId -> A.List A.JobId -> A.List A.JobId
addOptionalCandidate A.None xs = xs
addOptionalCandidate (A.Some j) xs = addCandidateOnce j xs

addRunnableCandidates :: A.List A.JobId -> A.List A.JobId -> A.List A.JobId
addRunnableCandidates runnable xs =
  case runnable of
    A.Nil -> xs
    A.Cons j runnable' -> addRunnableCandidates runnable' (addCandidateOnce j xs)

candidateRowFromRow :: A.AwkernelCapturedRow -> A.List A.JobId
candidateRowFromRow row =
  addOptionalCandidate
    (A.acr_dispatch_target row)
    (addRunnableCandidates
      (A.acr_runnable row)
      (addOptionalCandidate (A.acr_current row) A.Nil))

candidateTableFromRows :: A.List A.AwkernelCapturedRow -> A.List (A.List A.JobId)
candidateTableFromRows rows =
  case rows of
    A.Nil -> A.Nil
    A.Cons row rows' -> A.Cons (candidateRowFromRow row) (candidateTableFromRows rows')

jobListToText :: A.List A.JobId -> String
jobListToText jobs = "[" ++ go jobs ++ "]"
  where
    go A.Nil = ""
    go (A.Cons j A.Nil) = show (natToInteger j)
    go (A.Cons j rest) = show (natToInteger j) ++ "; " ++ go rest

candidateTableLines :: A.List (A.List A.JobId) -> [String]
candidateTableLines A.Nil = ["  []."]
candidateTableLines (A.Cons row A.Nil) = ["  [ " ++ jobListToText row, "  ]."]
candidateTableLines (A.Cons row rest) = ("  [ " ++ jobListToText row) : candidateTableTail rest
  where
    candidateTableTail A.Nil = ["  ]."]
    candidateTableTail (A.Cons row' A.Nil) = ["  ; " ++ jobListToText row', "  ]."]
    candidateTableTail (A.Cons row' rest') = ("  ; " ++ jobListToText row') : candidateTableTail rest'

renderCandidateTable :: A.List (A.List A.JobId) -> String
renderCandidateTable table =
  unlines
    ( [ "From Stdlib Require Import List Arith.PeanoNat."
      , "From RocqSched Require Import Foundation.Base."
      , "Import ListNotations."
      , ""
      , "Definition awk_generated_candidate_table : list (list JobId) :="
      ]
      ++ candidateTableLines table
    )

main :: IO ()
main = do
  args <- getArgs
  let (backend, rowsPath, outPath) = case args of
        (x:y:z:_) -> (x, y, z)
        _ -> ("backend", "", "")
  if null rowsPath || null outPath
    then do
      hPutStrLn stderr "backend: expected arguments <backend> <rows-file> <output-file>"
      exitFailure
    else do
      rowsInput <- readFile rowsPath
      case rowsFromLines (lines rowsInput) of
        Left err -> do
          hPutStrLn stderr (backend ++ ": failed to parse trace rows: " ++ err)
          exitFailure
        Right rows -> do
          let table = candidateTableFromRows rows
          case A.candidate_table_matches_rows rows table of
            A.False -> do
              hPutStrLn stderr (backend ++ ": candidate-table sanity check failed")
              exitFailure
            A.True -> do
              writeFile outPath (renderCandidateTable table)
              putStrLn (backend ++ ": generated candidate_table.v")
              exitSuccess
