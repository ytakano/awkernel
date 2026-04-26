module Main where

import qualified AwkernelHandoffAcceptance as A
import Data.Char (isDigit)
import System.Environment (getArgs)
import System.Exit (exitFailure, exitSuccess)
import System.IO (getContents, hPutStrLn, stderr)

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
rowFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField, _candidatePrefixCsv] =
  rowFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField]
rowFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField, _workerCurrentCsv, _workerNeedReschedCsv, _workerDispatchCsv] =
  rowFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField]
rowFromFields fields =
  Left ("expected 8, 9, or 11 TSV columns, got " ++ show (length fields) ++ " from " ++ show fields)

rowsFromLines :: [String] -> Either String (A.List A.AwkernelCapturedRow)
rowsFromLines [] = Right A.Nil
rowsFromLines (line:rest)
  | null line = rowsFromLines rest
  | otherwise = do
      row <- rowFromFields (splitOn '\t' line)
      rows <- rowsFromLines rest
      pure (A.Cons row rows)

main :: IO ()
main = do
  args <- getArgs
  let backend = case args of
        [] -> "backend"
        (x:_) -> x
  input <- getContents
  case rowsFromLines (lines input) of
    Left err -> do
      hPutStrLn stderr (backend ++ ": failed to parse trace rows: " ++ err)
      exitFailure
    Right rows ->
      case A.awk_handoff_accepts_rows rows of
        A.True -> do
          putStrLn (backend ++ ": acceptance checker accepted trace rows")
          exitSuccess
        A.False -> do
          hPutStrLn stderr (backend ++ ": acceptance checker rejected trace rows")
          exitFailure
