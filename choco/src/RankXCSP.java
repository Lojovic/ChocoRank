import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

import org.chocosolver.parser.xcsp.ChocoXCSP;
import org.chocosolver.parser.xcsp.XCSP;
import org.chocosolver.solver.constraints.Propagator;
import org.chocosolver.solver.constraints.PropagatorPriority;

public class RankXCSP {

    private static final String LIMIT = "00h020m00s";

    record ProbeResult(long runtimeMs, Map<String, Double> propagatorScores) {}
    record SolveResult(long runtimeMs, Map<String, Double> propagatorScores) {}
    record GatedPolicy(String unseenDecision, Map<String, String> classDecisions) {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args[0].equals("--help") || args[0].equals("-h")) {
            usage();
            return;
        }

        switch (args[0]) {
            case "--probe" -> runProbe(args);
            case "--run" -> runWithPriorities(args);
            case "--run-default" -> runDefault(args);
            case "--run-model" -> runWithRankModel(args);
            case "--run-gated-model" -> runWithGatedRankModel(args);
            case "--solve-default" -> printDefaultSolution(args);
            case "--solve-model" -> printRankModelSolution(args);
            case "--solve-gated-model" -> printGatedRankModelSolution(args);
            default -> {
                usage();
                throw new IllegalArgumentException("Unknown command: " + args[0]);
            }
        }
    }

    static void usage() {
        System.err.println("Usage:");
        System.err.println("  java -jar choco-rank.jar --probe <probe-json> <instance.xml>");
        System.err.println("  java -jar choco-rank.jar --run <result-json> <priorities-json> <instance.xml>");
        System.err.println("  java -jar choco-rank.jar --run-default <result-json> <instance.xml>");
        System.err.println("  java -jar choco-rank.jar --run-model <result-json> <project-root> <train-dir> <xcsp2feat> <instance.xml>");
        System.err.println("  java -jar choco-rank.jar --run-gated-model <result-json> <project-root> <train-dir> <xcsp2feat> <instance.xml> [class-policy.json]");
        System.err.println("  java -jar choco-rank.jar --solve-default <instance.xml>");
        System.err.println("  java -jar choco-rank.jar --solve-model <project-root> <train-dir> <xcsp2feat> <instance.xml>");
        System.err.println("  java -jar choco-rank.jar --solve-gated-model <project-root> <train-dir> <xcsp2feat> <instance.xml> [class-policy.json]");
    }

    static void runProbe(String[] args) throws Exception {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --probe <probe-json> <instance.xml>");
        }

        Path outJson = Paths.get(args[1]);
        Path instance = validateInstance(args[2]);

        ProbeResult result = probeInstance(instance);
        writeProbeResult(outJson, instance.getFileName().toString(), result);
        System.out.printf(Locale.US, "Probed: %s (%d propagators, %d ms)%n",
                instance.getFileName(), result.propagatorScores().size(), result.runtimeMs());
    }

    static void runWithPriorities(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --run <result-json> <priorities-json> <instance.xml>");
        }

        Path outJson = Paths.get(args[1]);
        Path prioritiesJson = Paths.get(args[2]);
        Path instance = validateInstance(args[3]);
        if (!Files.isRegularFile(prioritiesJson)) {
            throw new IllegalArgumentException("Expected a priorities JSON file: " + prioritiesJson);
        }

        System.out.println("Running: " + instance.getFileName());
        SolveResult result = solveInstance(instance, prioritiesJson);
        writeSolveResult(outJson, instance.getFileName().toString(), result);
        System.out.printf(Locale.US, "  recorded 01/01: %d ms, %d propagator scores%n",
                result.runtimeMs(), result.propagatorScores().size());
        System.out.println("Saved: " + outJson);
    }

    static void runDefault(String[] args) throws Exception {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --run-default <result-json> <instance.xml>");
        }

        Path outJson = Paths.get(args[1]);
        Path instance = validateInstance(args[2]);

        System.out.println("Running default Choco: " + instance.getFileName());
        SolveResult result = solveDefaultInstance(instance);
        writeSolveResult(outJson, instance.getFileName().toString(), result);
        System.out.printf(Locale.US, "  recorded 01/01: %d ms, %d propagator scores%n",
                result.runtimeMs(), result.propagatorScores().size());
        System.out.println("Saved: " + outJson);
    }

    static void runWithRankModel(String[] args) throws Exception {
        if (args.length != 6) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --run-model <result-json> <project-root> <train-dir> <xcsp2feat> <instance.xml>");
        }

        Path outJson = Paths.get(args[1]);
        Path rankDir = Paths.get(args[2]);
        Path trainDir = Paths.get(args[3]);
        Path featureExtractor = Paths.get(args[4]);
        Path instance = validateInstance(args[5]);
        validateDirectory(rankDir, "Rank directory");
        validateDirectory(trainDir, "Rank train_dir");
        if (!Files.isRegularFile(featureExtractor)) {
            throw new IllegalArgumentException("Expected xcsp2feat executable: " + featureExtractor);
        }

        Path outputDir = outputDirFor(outJson);
        Path priorityDir = outputDir.resolve("rank-priorities");
        Files.createDirectories(priorityDir);
        String instanceName = instance.getFileName().toString();
        Path probeJson = priorityDir.resolve(instanceName + ".probe.json");
        Path featureCsv = priorityDir.resolve(instanceName + ".features.csv");
        Path prioritiesJson = priorityDir.resolve(instanceName + ".priorities.json");

        System.out.println("Building model: " + instance.getFileName());
        BuiltModel builtModel = buildModel(instance);
        writeProbeResult(probeJson, instanceName,
                new ProbeResult(builtModel.buildMs(), propagatorScores()));
        System.out.printf(Locale.US, "  collected %d propagator classes in %d ms%n",
                propagatorScores().size(), builtModel.buildMs());

        runRankPrediction(
                rankDir,
                trainDir,
                featureExtractor,
                instance,
                probeJson,
                featureCsv,
                priorityDir);

        Propagator.applyModelPriorities(readModelPriorities(prioritiesJson));

        System.out.println("Running: " + instance.getFileName());
        long solveStart = System.nanoTime();
        try {
            builtModel.xcsp().configureSearch();
            builtModel.xcsp().solve();
        } finally {
            clearPropagatorStateFlags();
        }
        long solveMs = (System.nanoTime() - solveStart) / 1_000_000;
        long chocoMs = builtModel.buildMs() + solveMs;

        SolveResult result = new SolveResult(chocoMs, propagatorScores());
        writeSolveResult(outJson, instanceName, result);
        System.out.printf(Locale.US, "  recorded 01/01: %d ms, %d propagator scores%n",
                result.runtimeMs(), result.propagatorScores().size());
        System.out.println("Saved: " + outJson);
    }

    static void printRankModelSolution(String[] args) throws Exception {
        if (args.length != 5) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --solve-model <project-root> <train-dir> <xcsp2feat> <instance.xml>");
        }
        Path rankDir = Paths.get(args[1]);
        Path trainDir = Paths.get(args[2]);
        Path featureExtractor = Paths.get(args[3]);
        Path instance = validateInstance(args[4]);
        validateModelInputs(rankDir, trainDir, featureExtractor);

        Path priorityDir = Files.createTempDirectory("rank-priorities-");
        try {
            BuiltModel builtModel = prepareRankModel(
                    rankDir,
                    trainDir,
                    featureExtractor,
                    instance,
                    priorityDir,
                    true);
            printSolutionOnly(builtModel.xcsp());
        } finally {
            deleteRecursively(priorityDir);
            clearPropagatorStateFlags();
        }
    }

    static void printDefaultSolution(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --solve-default <instance.xml>");
        }
        Path instance = validateInstance(args[1]);
        printDefaultSolutionOnly(instance);
    }

    static void runWithGatedRankModel(String[] args) throws Exception {
        if (args.length != 6 && args.length != 7) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --run-gated-model <result-json> <project-root> <train-dir> <xcsp2feat> <instance.xml> [class-policy.json]");
        }

        Path outJson = Paths.get(args[1]);
        Path rankDir = Paths.get(args[2]);
        Path trainDir = Paths.get(args[3]);
        Path featureExtractor = Paths.get(args[4]);
        Path instance = validateInstance(args[5]);
        validateDirectory(rankDir, "Rank directory");
        validateDirectory(trainDir, "Rank train_dir");
        if (!Files.isRegularFile(featureExtractor)) {
            throw new IllegalArgumentException("Expected xcsp2feat executable: " + featureExtractor);
        }

        Path policyJson = args.length == 7 ? Paths.get(args[6]) : trainDir.resolve("class_policy.json");
        GatedPolicy policy = readGatedPolicy(policyJson);
        String problemClass = problemClassFor(instance);
        String decision = policy.classDecisions().get(problemClass);
        String reason = "policy";
        if (decision == null) {
            decision = policy.unseenDecision();
            reason = "unseen_class_fallback";
        }

        if (decision.equals("model")) {
            System.out.printf(Locale.US,
                    "Gated policy: class=%s decision=model reason=%s%n",
                    problemClass, reason);
            runWithRankModel(new String[]{
                    "--run-model",
                    outJson.toString(),
                    rankDir.toString(),
                    trainDir.toString(),
                    featureExtractor.toString(),
                    instance.toString()});
        } else if (decision.equals("default")) {
            System.out.printf(Locale.US,
                    "Gated policy: class=%s decision=default reason=%s%n",
                    problemClass, reason);
            System.out.println("Running default Choco: " + instance.getFileName());
            SolveResult result = solveDefaultInstance(instance);
            writeSolveResult(outJson, instance.getFileName().toString(), result);
            System.out.printf(Locale.US, "  recorded 01/01: %d ms, %d propagator scores%n",
                    result.runtimeMs(), result.propagatorScores().size());
            System.out.println("Saved: " + outJson);
        } else {
            throw new IllegalArgumentException(
                    "Invalid gated policy decision for " + problemClass + ": " + decision);
        }
    }

    static void printGatedRankModelSolution(String[] args) throws Exception {
        if (args.length != 5 && args.length != 6) {
            throw new IllegalArgumentException(
                    "Usage: java -jar choco-rank.jar --solve-gated-model <project-root> <train-dir> <xcsp2feat> <instance.xml> [class-policy.json]");
        }

        Path rankDir = Paths.get(args[1]);
        Path trainDir = Paths.get(args[2]);
        Path featureExtractor = Paths.get(args[3]);
        Path instance = validateInstance(args[4]);
        validateModelInputs(rankDir, trainDir, featureExtractor);

        Path policyJson = args.length == 6 ? Paths.get(args[5]) : trainDir.resolve("class_policy.json");
        GatedPolicy policy = readGatedPolicy(policyJson);
        String problemClass = problemClassFor(instance);
        String decision = policy.classDecisions().get(problemClass);
        if (decision == null) {
            decision = policy.unseenDecision();
        }

        if (decision.equals("model")) {
            printRankModelSolution(new String[]{
                    "--solve-model",
                    rankDir.toString(),
                    trainDir.toString(),
                    featureExtractor.toString(),
                    instance.toString()});
        } else if (decision.equals("default")) {
            printDefaultSolutionOnly(instance);
        } else {
            throw new IllegalArgumentException(
                    "Invalid gated policy decision for " + problemClass + ": " + decision);
        }
    }

    static Path validateInstance(String value) {
        Path instance = Paths.get(value);
        if (!Files.isRegularFile(instance) || !instance.toString().endsWith(".xml")) {
            throw new IllegalArgumentException("Expected an XCSP XML file: " + instance);
        }
        return instance;
    }

    static void validateDirectory(Path path, String description) {
        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("Expected " + description + ": " + path);
        }
    }

    static void validateModelInputs(Path rankDir, Path trainDir, Path featureExtractor) {
        validateDirectory(rankDir, "Rank directory");
        validateDirectory(trainDir, "Rank train_dir");
        if (!Files.isRegularFile(featureExtractor)) {
            throw new IllegalArgumentException("Expected xcsp2feat executable: " + featureExtractor);
        }
    }

    static Path outputDirFor(Path outJson) {
        Path parent = outJson.toAbsolutePath().getParent();
        return parent == null ? Paths.get(".").toAbsolutePath() : parent;
    }

    record BuiltModel(XCSP xcsp, long buildMs) {}

    static BuiltModel buildModel(Path instance) throws Exception {
        clearPropagatorState();
        long start = System.nanoTime();
        XCSP xcsp = new XCSP();
        if (!xcsp.setUp("-lvl", "SILENT", instance.toString())) {
            throw new IllegalArgumentException("Could not set up XCSP parser for " + instance);
        }
        xcsp.createSolver();
        xcsp.buildModel();
        long buildMs = (System.nanoTime() - start) / 1_000_000;
        return new BuiltModel(xcsp, buildMs);
    }

    static BuiltModel prepareRankModel(
            Path rankDir,
            Path trainDir,
            Path featureExtractor,
            Path instance,
            Path priorityDir,
            boolean quiet) throws Exception {
        Files.createDirectories(priorityDir);
        String instanceName = instance.getFileName().toString();
        Path probeJson = priorityDir.resolve(instanceName + ".probe.json");
        Path featureCsv = priorityDir.resolve(instanceName + ".features.csv");
        Path prioritiesJson = priorityDir.resolve(instanceName + ".priorities.json");

        BuiltModel builtModel = buildModel(instance);
        writeProbeResult(probeJson, instanceName,
                new ProbeResult(builtModel.buildMs(), propagatorScores()));

        runRankPrediction(
                rankDir,
                trainDir,
                featureExtractor,
                instance,
                probeJson,
                featureCsv,
                priorityDir,
                quiet);

        Propagator.applyModelPriorities(readModelPriorities(prioritiesJson));
        return builtModel;
    }

    static void runRankPrediction(
            Path rankDir,
            Path trainDir,
            Path featureExtractor,
            Path instance,
            Path probeJson,
            Path featureCsv,
            Path priorityDir) throws IOException, InterruptedException {
        runRankPrediction(
                rankDir,
                trainDir,
                featureExtractor,
                instance,
                probeJson,
                featureCsv,
                priorityDir,
                false);
    }

    static void runRankPrediction(
            Path rankDir,
            Path trainDir,
            Path featureExtractor,
            Path instance,
            Path probeJson,
            Path featureCsv,
            Path priorityDir,
            boolean quiet) throws IOException, InterruptedException {
        Path matplotlibDir = priorityDir.resolve(".matplotlib");
        Files.createDirectories(matplotlibDir);

        String python = System.getenv().getOrDefault("RANK_PYTHON", "python3");
        List<String> command = List.of(
                python,
                "src/choco_priority_pipeline.py",
                "--train_dir", trainDir.toAbsolutePath().toString(),
                "--input_json", probeJson.toAbsolutePath().toString(),
                "--instance_file", instance.toAbsolutePath().toString(),
                "--feature_extractor", featureExtractor.toAbsolutePath().toString(),
                "--feature_output", featureCsv.toAbsolutePath().toString(),
                "--output_dir", priorityDir.toAbsolutePath().toString());

        ProcessBuilder builder = new ProcessBuilder(command);
        builder.directory(rankDir.toFile());
        if (quiet) {
            builder.redirectOutput(ProcessBuilder.Redirect.DISCARD);
            builder.redirectError(ProcessBuilder.Redirect.INHERIT);
        } else {
            builder.inheritIO();
        }
        builder.environment().putIfAbsent("TF_CPP_MIN_LOG_LEVEL", "3");
        builder.environment().putIfAbsent("MPLCONFIGDIR", matplotlibDir.toString());
        int exitCode = builder.start().waitFor();
        if (exitCode != 0) {
            throw new IllegalStateException("Rank prediction failed with exit code " + exitCode);
        }
    }

    static ProbeResult probeInstance(Path instance) throws Exception {
        long start = System.nanoTime();
        try {
            buildModel(instance);
        } finally {
            clearPropagatorStateFlags();
        }
        long runtimeMs = (System.nanoTime() - start) / 1_000_000;
        return new ProbeResult(runtimeMs, propagatorScores());
    }

    static SolveResult solveInstance(Path instance, Path prioritiesJson) throws Exception {
        clearPropagatorState();
        Propagator.MODEL_PRIORITIES_ENABLED = true;
        Propagator.setModelPriorities(readModelPriorities(prioritiesJson));

        long start = System.nanoTime();
        try {
            ChocoXCSP.main(new String[]{"-lvl", "SILENT", "-limit", LIMIT, instance.toString()});
        } finally {
            clearPropagatorStateFlags();
        }
        long runtimeMs = (System.nanoTime() - start) / 1_000_000;
        return new SolveResult(runtimeMs, propagatorScores());
    }

    static void printSolutionOnly(XCSP xcsp) throws Exception {
        try {
            xcsp.configureSearch();
            xcsp.solve();
            System.out.print(solutionText(xcsp));
        } finally {
            clearPropagatorStateFlags();
        }
    }

    static void printDefaultSolutionOnly(Path instance) throws Exception {
        clearPropagatorState();
        XCSP xcsp = new XCSP();
        if (!xcsp.setUp("-lvl", "SILENT", instance.toString())) {
            throw new IllegalArgumentException("Could not set up XCSP parser for " + instance);
        }
        try {
            xcsp.createSolver();
            xcsp.buildModel();
            printSolutionOnly(xcsp);
        } finally {
            clearPropagatorStateFlags();
        }
    }

    static String solutionText(XCSP xcsp) throws Exception {
        Field outputField = XCSP.class.getDeclaredField("output");
        outputField.setAccessible(true);
        Object value = outputField.get(xcsp);
        return value == null ? "" : value.toString();
    }

    static void deleteRecursively(Path path) throws IOException {
        if (path == null || !Files.exists(path)) {
            return;
        }
        try (var stream = Files.walk(path)) {
            for (Path item : stream.sorted((left, right) -> right.compareTo(left)).toList()) {
                Files.deleteIfExists(item);
            }
        }
    }

    static SolveResult solveDefaultInstance(Path instance) throws Exception {
        clearPropagatorState();
        long start = System.nanoTime();
        try {
            ChocoXCSP.main(new String[]{"-lvl", "SILENT", "-limit", LIMIT, instance.toString()});
        } finally {
            clearPropagatorStateFlags();
        }
        long runtimeMs = (System.nanoTime() - start) / 1_000_000;
        return new SolveResult(runtimeMs, propagatorScores());
    }

    static void clearPropagatorState() {
        Propagator.clearPriorities();
        Propagator.SEED = 0;
        Propagator.SEED_OFFSET = 0L;
        clearPropagatorStateFlags();
    }

    static void clearPropagatorStateFlags() {
        Propagator.RANDOM_PRIORITIES = false;
        Propagator.MODEL_PRIORITIES_ENABLED = false;
    }

    static void writeProbeResult(Path outJson, String instance, ProbeResult result) throws IOException {
        Path parent = outJson.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        List<String> propagators = new ArrayList<>(result.propagatorScores().keySet());
        Collections.sort(propagators);

        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
                outJson,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING))) {
            pw.println("{");
            pw.println("  \"instance\": \"" + jsonEscape(instance) + "\",");
            pw.println("  \"runtime_ms\": " + result.runtimeMs() + ",");
            pw.println("  \"propagators\": [");
            for (int index = 0; index < propagators.size(); index++) {
                pw.print("    \"" + jsonEscape(propagators.get(index)) + "\"");
                pw.println(index < propagators.size() - 1 ? "," : "");
            }
            pw.println("  ],");
            writeScoresObject(pw, result.propagatorScores(), "  ");
            pw.println("}");
        }
    }

    static void writeSolveResult(Path outJson, String instance, SolveResult result) throws IOException {
        Path parent = outJson.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(
                outJson,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING))) {
            pw.println("[");
            pw.println("  {");
            pw.println("    \"instance\": \"" + jsonEscape(instance) + "\",");
            pw.println("    \"runtime_ms\": " + result.runtimeMs() + ",");
            writeScoresObject(pw, result.propagatorScores(), "    ");
            pw.println("  }");
            pw.println("]");
        }
    }

    static void writeScoresObject(PrintWriter pw, Map<String, Double> scores, String indent) {
        pw.println(indent + "\"propagator_scores\": {");
        int index = 0;
        for (Map.Entry<String, Double> entry : scores.entrySet()) {
            pw.print(indent + "  \"" + jsonEscape(entry.getKey()) + "\": " + entry.getValue());
            pw.println(index < scores.size() - 1 ? "," : "");
            index++;
        }
        pw.println(indent + "}");
    }

    static Map<String, Double> propagatorScores() {
        Map<String, Double> scores = new TreeMap<>();
        for (Map.Entry<Class<?>, Double> entry : Propagator.CLASS_SCORES.entrySet()) {
            scores.put(entry.getKey().getSimpleName(), entry.getValue());
        }
        return scores;
    }

    static Map<String, PropagatorPriority> readModelPriorities(Path prioritiesJson) throws IOException {
        String json = Files.readString(prioritiesJson);
        int prioritiesField = json.indexOf("\"priorities\"");
        if (prioritiesField < 0) {
            throw new IllegalArgumentException("Missing priorities object in " + prioritiesJson);
        }
        int open = json.indexOf('{', prioritiesField);
        int close = findMatchingBrace(json, open);
        if (open < 0 || close < 0) {
            throw new IllegalArgumentException("Invalid priorities object in " + prioritiesJson);
        }

        Map<String, PropagatorPriority> priorities = new LinkedHashMap<>();
        int cursor = open + 1;
        while (cursor < close) {
            int keyStart = json.indexOf('"', cursor);
            if (keyStart < 0 || keyStart >= close) {
                break;
            }
            int keyEnd = findStringEnd(json, keyStart);
            int colon = json.indexOf(':', keyEnd + 1);
            int valueStart = json.indexOf('"', colon + 1);
            int valueEnd = findStringEnd(json, valueStart);
            if (keyEnd < 0 || colon < 0 || valueStart < 0 || valueEnd < 0 || valueEnd > close) {
                throw new IllegalArgumentException("Invalid priority entry in " + prioritiesJson);
            }

            String propagator = unescapeJsonString(json.substring(keyStart + 1, keyEnd));
            String priority = unescapeJsonString(json.substring(valueStart + 1, valueEnd));
            priorities.put(propagator, PropagatorPriority.valueOf(priority));
            cursor = valueEnd + 1;
        }
        return priorities;
    }

    static GatedPolicy readGatedPolicy(Path policyJson) throws IOException {
        if (!Files.isRegularFile(policyJson)) {
            return new GatedPolicy("model", Collections.emptyMap());
        }
        String json = Files.readString(policyJson);
        String unseenDecision = readStringField(json, "default_decision_for_unseen_class", "model");
        if (!unseenDecision.equals("model") && !unseenDecision.equals("default")) {
            throw new IllegalArgumentException(
                    "Invalid default_decision_for_unseen_class in " + policyJson + ": " + unseenDecision);
        }

        int classesField = json.indexOf("\"classes\"");
        if (classesField < 0) {
            throw new IllegalArgumentException("Missing classes object in " + policyJson);
        }
        int open = json.indexOf('{', classesField);
        int close = findMatchingBrace(json, open);
        if (open < 0 || close < 0) {
            throw new IllegalArgumentException("Invalid classes object in " + policyJson);
        }

        Map<String, String> decisions = new LinkedHashMap<>();
        int cursor = open + 1;
        while (cursor < close) {
            int keyStart = json.indexOf('"', cursor);
            if (keyStart < 0 || keyStart >= close) {
                break;
            }
            int keyEnd = findStringEnd(json, keyStart);
            int colon = json.indexOf(':', keyEnd + 1);
            int objectStart = json.indexOf('{', colon + 1);
            if (keyEnd < 0 || colon < 0 || objectStart < 0 || objectStart >= close) {
                throw new IllegalArgumentException("Invalid class policy entry in " + policyJson);
            }
            int objectEnd = findMatchingBrace(json, objectStart);
            if (objectEnd < 0 || objectEnd > close) {
                throw new IllegalArgumentException("Invalid class policy object in " + policyJson);
            }

            String problemClass = unescapeJsonString(json.substring(keyStart + 1, keyEnd));
            String entryJson = json.substring(objectStart, objectEnd + 1);
            String decision = readStringField(entryJson, "decision", unseenDecision);
            if (!decision.equals("model") && !decision.equals("default")) {
                throw new IllegalArgumentException(
                        "Invalid decision for class " + problemClass + " in " + policyJson + ": " + decision);
            }
            decisions.put(problemClass, decision);
            cursor = objectEnd + 1;
        }
        return new GatedPolicy(unseenDecision, decisions);
    }

    static String readStringField(String json, String fieldName, String defaultValue) {
        int field = json.indexOf("\"" + fieldName + "\"");
        if (field < 0) {
            return defaultValue;
        }
        int colon = json.indexOf(':', field);
        int valueStart = json.indexOf('"', colon + 1);
        int valueEnd = findStringEnd(json, valueStart);
        if (colon < 0 || valueStart < 0 || valueEnd < 0) {
            return defaultValue;
        }
        return unescapeJsonString(json.substring(valueStart + 1, valueEnd));
    }

    static String problemClassFor(Path instance) {
        String filename = instance.getFileName().toString();
        if (filename.endsWith(".xml")) {
            filename = filename.substring(0, filename.length() - 4);
        }
        int dash = filename.indexOf('-');
        if (dash > 0) {
            return filename.substring(0, dash);
        }
        return filename;
    }

    static int findMatchingBrace(String json, int openBrace) {
        if (openBrace < 0 || openBrace >= json.length() || json.charAt(openBrace) != '{') {
            return -1;
        }

        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        for (int i = openBrace; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = inString;
            } else if (c == '"') {
                inString = !inString;
            } else if (!inString && c == '{') {
                depth++;
            } else if (!inString && c == '}') {
                depth--;
                if (depth == 0) {
                    return i;
                }
            }
        }
        return -1;
    }

    static int findStringEnd(String json, int quote) {
        if (quote < 0 || quote >= json.length() || json.charAt(quote) != '"') {
            return -1;
        }
        boolean escaped = false;
        for (int i = quote + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                return i;
            }
        }
        return -1;
    }

    static String unescapeJsonString(String text) {
        StringBuilder result = new StringBuilder(text.length());
        boolean escaped = false;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (escaped) {
                result.append(unescapeJsonChar(c));
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else {
                result.append(c);
            }
        }
        return result.toString();
    }

    static char unescapeJsonChar(char c) {
        return switch (c) {
            case '"' -> '"';
            case '\\' -> '\\';
            case '/' -> '/';
            case 'b' -> '\b';
            case 'f' -> '\f';
            case 'n' -> '\n';
            case 'r' -> '\r';
            case 't' -> '\t';
            default -> c;
        };
    }

    static String jsonEscape(String text) {
        StringBuilder escaped = new StringBuilder(text.length() + 16);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '"' -> escaped.append("\\\"");
                case '\\' -> escaped.append("\\\\");
                case '\b' -> escaped.append("\\b");
                case '\f' -> escaped.append("\\f");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (c < 0x20) {
                        escaped.append(String.format("\\u%04x", (int) c));
                    } else {
                        escaped.append(c);
                    }
                }
            }
        }
        return escaped.toString();
    }
}
