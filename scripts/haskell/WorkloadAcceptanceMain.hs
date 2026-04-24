module Main where

import qualified AwkernelWorkloadAcceptance as A
import Data.Char (isDigit)
import Data.List (intercalate)
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

natToInt :: A.Nat -> Int
natToInt A.O = 0
natToInt (A.S n) = 1 + natToInt n

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

optionListFromCsv :: String -> Either String (A.List (A.Option A.JobId))
optionListFromCsv "" = Right A.Nil
optionListFromCsv csv = listFromFields (splitOn ',' csv)
  where
    listFromFields [] = Right A.Nil
    listFromFields (x:xs) = do
      headNat <- optionNatFromField x
      tailNats <- listFromFields xs
      pure (A.Cons headNat tailNats)

boolListFromCsv :: String -> Either String (A.List A.Bool)
boolListFromCsv "" = Right A.Nil
boolListFromCsv csv = listFromFields (splitOn ',' csv)
  where
    listFromFields [] = Right A.Nil
    listFromFields (x:xs) = do
      headBool <- boolFromField x
      tailBools <- listFromFields xs
      pure (A.Cons headBool tailBools)

eventFromFields :: String -> String -> String -> Either String A.OpEvent
eventFromFields "Wakeup" a "-" = A.EvWakeup <$> natFromField a
eventFromFields "RequestResched" a "-" = A.EvRequestResched <$> natFromField a
eventFromFields "HandleResched" a "-" = A.EvHandleResched <$> natFromField a
eventFromFields "Choose" a b = A.EvChoose <$> natFromField a <*> natFromField b
eventFromFields "Dispatch" a b = A.EvDispatch <$> natFromField a <*> natFromField b
eventFromFields "Complete" a "-" = A.EvComplete <$> natFromField a
eventFromFields "JoinTargetReady" a "-" = A.EvJoinTargetReady <$> natFromField a
eventFromFields "Stutter" "-" "-" = Right A.EvStutter
eventFromFields tag _ _ = Left ("unsupported event fields: " ++ show tag)

schedTraceEntryFromFields :: [String] -> Either String A.AwkernelSchedTraceEntry
schedTraceEntryFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField] =
  schedTraceEntryFromCoreFields
    cpuField eventTag eventA eventB currentField runnableCsv needReschedField dispatchField
    currentField needReschedField dispatchField
schedTraceEntryFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField, candidatePrefixCsv] = do
  _candidatePrefix <- listFromCsv candidatePrefixCsv
  schedTraceEntryFromCoreFields
    cpuField eventTag eventA eventB currentField runnableCsv needReschedField dispatchField
    currentField needReschedField dispatchField
schedTraceEntryFromFields [cpuField, eventTag, eventA, eventB, currentField, runnableCsv, needReschedField, dispatchField, workerCurrentCsv, workerNeedReschedCsv, workerDispatchCsv] =
  schedTraceEntryFromCoreFields
    cpuField eventTag eventA eventB currentField runnableCsv needReschedField dispatchField
    workerCurrentCsv workerNeedReschedCsv workerDispatchCsv
schedTraceEntryFromFields fields =
  Left ("expected 8, 9, or 11 TSV columns, got " ++ show (length fields) ++ " from " ++ show fields)

schedTraceEntryFromCoreFields :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Either String A.AwkernelSchedTraceEntry
schedTraceEntryFromCoreFields cpuField eventTag eventA eventB currentField runnableCsv needReschedField dispatchField workerCurrentCsv workerNeedReschedCsv workerDispatchCsv = do
  cpu <- natFromField cpuField
  event <- eventFromFields eventTag eventA eventB
  current <- optionNatFromField currentField
  runnable <- listFromCsv runnableCsv
  needResched <- boolFromField needReschedField
  dispatch <- optionNatFromField dispatchField
  workerCurrent <- optionListFromCsv workerCurrentCsv
  workerNeedResched <- boolListFromCsv workerNeedReschedCsv
  workerDispatch <- optionListFromCsv workerDispatchCsv
  pure (A.MkAwkernelSchedTraceEntry cpu event current runnable needResched dispatch workerCurrent workerNeedResched workerDispatch)

schedTraceFromLines :: Int -> [String] -> Either (Int, String) (A.List A.AwkernelSchedTraceEntry)
schedTraceFromLines _ [] = Right A.Nil
schedTraceFromLines index (line:rest)
  | null line = schedTraceFromLines (index + 1) rest
  | otherwise = do
      entry <- either (Left . (,) index) Right (schedTraceEntryFromFields (splitOn '\t' line))
      schedTrace <- schedTraceFromLines (index + 1) rest
      pure (A.Cons entry schedTrace)

taskTraceKindFromField :: String -> Either String A.AwkernelTaskTraceKind
taskTraceKindFromField "Spawn" = Right A.LkSpawn
taskTraceKindFromField "Runnable" = Right A.LkRunnable
taskTraceKindFromField "Choose" = Right A.LkChoose
taskTraceKindFromField "Dispatch" = Right A.LkDispatch
taskTraceKindFromField "Sleep" = Right A.LkSleep
taskTraceKindFromField "JoinWait" = Right A.LkJoinWait
taskTraceKindFromField "JoinTargetReady" = Right A.LkJoinTargetReady
taskTraceKindFromField "Complete" = Right A.LkComplete
taskTraceKindFromField field = Left ("unsupported task_trace kind: " ++ show field)

taskTraceEntryFromFields :: [String] -> Either String A.AwkernelTaskTraceEntry
taskTraceEntryFromFields [kindField, subjectField, relatedField] = do
  kind <- taskTraceKindFromField kindField
  subject <- natFromField subjectField
  related <- optionNatFromField relatedField
  pure (A.MkAwkernelTaskTraceEntry kind subject related)
taskTraceEntryFromFields fields =
  Left ("expected 3 TSV task_trace columns, got " ++ show (length fields) ++ " from " ++ show fields)

taskTraceFromLines :: Int -> [String] -> Either (Int, String) (A.List A.AwkernelTaskTraceEntry)
taskTraceFromLines _ [] = Right A.Nil
taskTraceFromLines index (line:rest)
  | null line = taskTraceFromLines (index + 1) rest
  | otherwise = do
      record <- either (Left . (,) index) Right (taskTraceEntryFromFields (splitOn '\t' line))
      records <- taskTraceFromLines (index + 1) rest
      pure (A.Cons record records)

data Diagnostic = Diagnostic
  { accepted :: Bool
  , kind :: String
  , message :: String
  , schedTraceIndex :: Maybe Int
  , taskTraceIndex :: Maybe Int
  , logLineBegin :: Maybe Int
  , logLineEnd :: Maybe Int
  , backendLabel :: String
  , scenarioLabel :: Maybe String
  }

jsonEscape :: String -> String
jsonEscape = concatMap escapeChar
  where
    escapeChar '"' = "\\\""
    escapeChar '\\' = "\\\\"
    escapeChar '\n' = "\\n"
    escapeChar '\r' = "\\r"
    escapeChar '\t' = "\\t"
    escapeChar c = [c]

jsonField :: String -> String -> String
jsonField key value = "\"" ++ key ++ "\":" ++ value

jsonString :: String -> String
jsonString s = "\"" ++ jsonEscape s ++ "\""

jsonMaybeInt :: Maybe Int -> String
jsonMaybeInt Nothing = "null"
jsonMaybeInt (Just n) = show n

jsonMaybeString :: Maybe String -> String
jsonMaybeString Nothing = "null"
jsonMaybeString (Just s) = jsonString s

renderDiagnostic :: Diagnostic -> String
renderDiagnostic diag =
  "{" ++ intercalate "," fields ++ "}"
  where
    fields =
      [ jsonField "accepted" (if accepted diag then "true" else "false")
      , jsonField "backend" (jsonString (backendLabel diag))
      , jsonField "scenario" (jsonMaybeString (scenarioLabel diag))
      , jsonField "kind" (jsonString (kind diag))
      , jsonField "message" (jsonString (message diag))
      , jsonField "sched_trace_index" (jsonMaybeInt (schedTraceIndex diag))
      , jsonField "task_trace_index" (jsonMaybeInt (taskTraceIndex diag))
      , jsonField "log_line_begin" (jsonMaybeInt (logLineBegin diag))
      , jsonField "log_line_end" (jsonMaybeInt (logLineEnd diag))
      ]

emitDiagnostic :: Diagnostic -> IO ()
emitDiagnostic diag = do
  putStrLn (renderDiagnostic diag)
  let label = case scenarioLabel diag of
        Nothing -> backendLabel diag
        Just s -> backendLabel diag ++ "-" ++ s
      status = if accepted diag then "accepted" else "rejected"
  hPutStrLn stderr (label ++ ": " ++ status ++ ": " ++ message diag)

mkSuccess :: String -> Maybe String -> Diagnostic
mkSuccess backend scenario =
  Diagnostic
    { accepted = True
    , kind = "accepted"
    , message = "workload acceptance accepted the emitted task_trace/sched_trace pair under the logical top-1 GlobalFIFO worker schedule"
    , schedTraceIndex = Nothing
    , taskTraceIndex = Nothing
    , logLineBegin = Nothing
    , logLineEnd = Nothing
    , backendLabel = backend
    , scenarioLabel = scenario
    }

mkFailure :: String -> Maybe String -> String -> String -> Maybe Int -> Maybe Int -> Diagnostic
mkFailure backend scenario diagKind diagMessage schedTraceIx taskTraceIx =
  Diagnostic
    { accepted = False
    , kind = diagKind
    , message = diagMessage
    , schedTraceIndex = schedTraceIx
    , taskTraceIndex = taskTraceIx
    , logLineBegin = Nothing
    , logLineEnd = Nothing
    , backendLabel = backend
    , scenarioLabel = scenario
    }

main :: IO ()
main = do
  args <- getArgs
  let (backend, scenarioRaw, schedTracePath, taskTracePath) = case args of
        (w:x:y:z:_) -> (w, x, y, z)
        _ -> ("backend", "-", "", "")
      scenario = if scenarioRaw == "-" then Nothing else Just scenarioRaw
  if null schedTracePath || null taskTracePath
    then do
      emitDiagnostic
        (mkFailure backend scenario "internal-checker-error"
          "expected arguments <backend> <scenario-or--> <sched-trace-file> <task-trace-file>"
          Nothing Nothing)
      exitFailure
    else do
      schedTraceInput <- readFile schedTracePath
      taskTraceInput <- readFile taskTracePath
      case schedTraceFromLines 0 (lines schedTraceInput) of
        Left (idx, err) -> do
          emitDiagnostic
            (mkFailure backend scenario "sched-trace-parse-failure"
              ("failed to parse extracted sched_trace: " ++ err)
              (Just idx) Nothing)
          exitFailure
        Right schedTrace ->
          case taskTraceFromLines 0 (lines taskTraceInput) of
            Left (idx, err) -> do
              emitDiagnostic
                (mkFailure backend scenario "task-trace-parse-failure"
                  ("failed to parse extracted task_trace: " ++ err)
                  Nothing (Just idx))
              exitFailure
            Right taskTrace ->
              case A.awk_workload_accepts_sched_trace taskTrace schedTrace of
                A.False -> do
                  emitDiagnostic
                    (mkFailure backend scenario "workload-family-rejection"
                      "workload acceptance rejected the emitted task_trace/sched_trace pair"
                      Nothing Nothing)
                  exitFailure
                A.True ->
                  case A.first_non_scheduler_relation_sched_trace_index taskTrace schedTrace of
                    A.None -> do
                      emitDiagnostic (mkSuccess backend scenario)
                      exitSuccess
                    A.Some relationIdx ->
                      case A.first_non_fifo_sched_trace_index schedTrace of
                        A.Some fifoIdx
                          | natToInt fifoIdx == natToInt relationIdx -> do
                              emitDiagnostic
                                (mkFailure backend scenario "global-fifo-rejection"
                                  "the emitted sched_trace violates the local GlobalFIFO choose-order check for the logical top-1 worker schedule"
                                  (Just (natToInt fifoIdx)) Nothing)
                              exitFailure
                        _ -> do
                          emitDiagnostic
                            (mkFailure backend scenario "scheduler-relation-rejection"
                              "the emitted sched_trace violates the extracted GlobalFIFO scheduler-relation check for the logical top-1 worker schedule"
                              (Just (natToInt relationIdx)) Nothing)
                          exitFailure
