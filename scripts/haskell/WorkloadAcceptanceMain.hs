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

lifecycleKindFromField :: String -> Either String A.TaskLifecycleKind
lifecycleKindFromField "Spawn" = Right A.LkSpawn
lifecycleKindFromField "Runnable" = Right A.LkRunnable
lifecycleKindFromField "Choose" = Right A.LkChoose
lifecycleKindFromField "Dispatch" = Right A.LkDispatch
lifecycleKindFromField "Sleep" = Right A.LkSleep
lifecycleKindFromField "JoinWait" = Right A.LkJoinWait
lifecycleKindFromField "Complete" = Right A.LkComplete
lifecycleKindFromField field = Left ("unsupported lifecycle kind: " ++ show field)

lifecycleRecordFromFields :: [String] -> Either String A.TaskLifecycleRecord
lifecycleRecordFromFields [kindField, subjectField, relatedField] = do
  kind <- lifecycleKindFromField kindField
  subject <- natFromField subjectField
  related <- optionNatFromField relatedField
  pure (A.MkTaskLifecycleRecord kind subject related)
lifecycleRecordFromFields fields =
  Left ("expected 3 TSV lifecycle columns, got " ++ show (length fields) ++ " from " ++ show fields)

lifecycleFromLines :: [String] -> Either String (A.List A.TaskLifecycleRecord)
lifecycleFromLines [] = Right A.Nil
lifecycleFromLines (line:rest)
  | null line = lifecycleFromLines rest
  | otherwise = do
      record <- lifecycleRecordFromFields (splitOn '\t' line)
      records <- lifecycleFromLines rest
      pure (A.Cons record records)

main :: IO ()
main = do
  args <- getArgs
  let (backend, rowsPath, lifecyclePath) = case args of
        (x:y:z:_) -> (x, y, z)
        _ -> ("backend", "", "")
  if null rowsPath || null lifecyclePath
    then do
      hPutStrLn stderr "backend: expected arguments <backend> <rows-file> <lifecycle-file>"
      exitFailure
    else do
      rowsInput <- readFile rowsPath
      lifecycleInput <- readFile lifecyclePath
      case rowsFromLines (lines rowsInput) of
        Left err -> do
          hPutStrLn stderr (backend ++ ": failed to parse trace rows: " ++ err)
          exitFailure
        Right rows ->
          case lifecycleFromLines (lines lifecycleInput) of
            Left err -> do
              hPutStrLn stderr (backend ++ ": failed to parse task lifecycle: " ++ err)
              exitFailure
            Right lifecycle ->
              case A.awk_workload_accepts_trace lifecycle rows of
                A.True -> do
                  putStrLn (backend ++ ": acceptance checker accepted workload trace")
                  exitSuccess
                A.False -> do
                  hPutStrLn stderr (backend ++ ": acceptance checker rejected workload trace")
                  exitFailure
