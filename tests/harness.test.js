import { assert, assertClose, assertArrayClose, runSuite } from "./harness.js";

await runSuite("harness self-test", {
  "assert passes on truthy": () => assert(true),
  "assertClose within tolerance": () => assertClose(1.000001, 1.0, 1e-4),
  "assertArrayClose matches vectors": () =>
    assertArrayClose([1, 2, 3], [1, 2, 3]),
});

console.log("\nAll harness tests passed.");
