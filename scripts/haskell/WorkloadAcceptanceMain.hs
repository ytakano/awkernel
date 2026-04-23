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

rowsFromLines :: Int -> [String] -> Either (Int, String) (A.List A.AwkernelCapturedRow)
rowsFromLines _ [] = Right A.Nil
rowsFromLines index (line:rest)
  | null line = rowsFromLines (index + 1) rest
  | otherwise = do
      row <- either (Left . (,) index) Right (rowFromFields (splitOn '\t' line))
      rows <- rowsFromLines (index + 1) rest
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

lifecycleFromLines :: Int -> [String] -> Either (Int, String) (A.List A.TaskLifecycleRecord)
lifecycleFromLines _ [] = Right A.Nil
lifecycleFromLines index (line:rest)
  | null line = lifecycleFromLines (index + 1) rest
  | otherwise = do
      record <- either (Left . (,) index) Right (lifecycleRecordFromFields (splitOn '\t' line))
      records <- lifecycleFromLines (index + 1) rest
      pure (A.Cons record records)

data Diagnostic = Diagnostic
  { accepted :: Bool
  , kind :: String
  , message :: String
  , rowIndex :: Maybe Int
  , lifecycleIndex :: Maybe Int
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
      , jsonField "row_index" (jsonMaybeInt (rowIndex diag))
      , jsonField "lifecycle_index" (jsonMaybeInt (lifecycleIndex diag))
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
    , message = "workload acceptance accepted the emitted lifecycle/rows trace"
    , rowIndex = Nothing
    , lifecycleIndex = Nothing
    , backendLabel = backend
    , scenarioLabel = scenario
    }

mkFailure :: String -> Maybe String -> String -> String -> Maybe Int -> Maybe Int -> Diagnostic
mkFailure backend scenario diagKind diagMessage rowIx lifecycleIx =
  Diagnostic
    { accepted = False
    , kind = diagKind
    , message = diagMessage
    , rowIndex = rowIx
    , lifecycleIndex = lifecycleIx
    , backendLabel = backend
    , scenarioLabel = scenario
    }

main :: IO ()
main = do
  args <- getArgs
  let (backend, scenarioRaw, rowsPath, lifecyclePath) = case args of
        (w:x:y:z:_) -> (w, x, y, z)
        _ -> ("backend", "-", "", "")
      scenario = if scenarioRaw == "-" then Nothing else Just scenarioRaw
  if null rowsPath || null lifecyclePath
    then do
      emitDiagnostic
        (mkFailure backend scenario "internal-checker-error"
          "expected arguments <backend> <scenario-or--> <rows-file> <lifecycle-file>"
          Nothing Nothing)
      exitFailure
    else do
      rowsInput <- readFile rowsPath
      lifecycleInput <- readFile lifecyclePath
      case rowsFromLines 0 (lines rowsInput) of
        Left (idx, err) -> do
          emitDiagnostic
            (mkFailure backend scenario "rows-parse-failure"
              ("failed to parse extracted trace rows: " ++ err)
              (Just idx) Nothing)
          exitFailure
        Right rows ->
          case lifecycleFromLines 0 (lines lifecycleInput) of
            Left (idx, err) -> do
              emitDiagnostic
                (mkFailure backend scenario "lifecycle-parse-failure"
                  ("failed to parse extracted task lifecycle: " ++ err)
                  Nothing (Just idx))
              exitFailure
            Right lifecycle ->
              case A.awk_workload_accepts_trace lifecycle rows of
                A.True -> do
                  emitDiagnostic (mkSuccess backend scenario)
                  exitSuccess
                A.False -> do
                  emitDiagnostic
                    (mkFailure backend scenario "workload-family-rejection"
                      "workload acceptance rejected the emitted lifecycle/rows trace"
                      Nothing Nothing)
                  exitFailure
