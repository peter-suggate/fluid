import {
  buildSvoShippedSceneCoverage,
  canonicalSvoSceneCoverage,
} from "../lib/svo-scene-coverage";

const report = buildSvoShippedSceneCoverage();
const output = process.argv.includes("--compact")
  ? canonicalSvoSceneCoverage(report)
  : JSON.stringify(report, null, 2);

process.stdout.write(`${output}\n`);
