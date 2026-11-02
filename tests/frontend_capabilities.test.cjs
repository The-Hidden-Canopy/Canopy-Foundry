const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(root, "frontend/app.js"), "utf8");

function consoleHarness() {
  const nodes = new Map();
  const makeNode = () => ({
    children: [], style: {}, dataset: {}, value: "",
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener() {},
  });
  const document = {
    documentElement: { dataset: {} },
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, makeNode());
      return nodes.get(id);
    },
    querySelector(selector) { return selector.startsWith("[data-view-panel=") ? makeNode() : null; },
    querySelectorAll() { return []; },
    createElement: makeNode,
  };
  const context = vm.createContext({ document, window: { setInterval() {} } });
  vm.runInContext(source, context, { filename: "app.js" });
  return {
    nodes, context,
    view(doc, sourceKind = "file") {
      context.selected = doc;
      context.sourceKind = sourceKind;
      return JSON.parse(JSON.stringify(vm.runInContext("capabilityViewForDocument(selected, sourceKind)", context)));
    },
    async load(doc) {
      document.getElementById("request-file").files = [{ name: "inventory.json", size: 100, text: async () => JSON.stringify(doc) }];
      await vm.runInContext("loadSelectedFile()", context);
    },
  };
}

const nativeInventory = (entries) => ({
  schema: "canopy-foundry-runtime-capabilities.v1",
  source: "build_inventory",
  capabilities: entries,
});
const entry = (id, state = "BUILD_DISABLED") => ({ id, state, evidence: "fixture declaration", evidence_tier: "SOURCE_ONLY" });

test("empty, example, and request documents do not invent capability rows", () => {
  const harness = consoleHarness();
  assert.equal(harness.nodes.get("capability-rows").children.length, 0);
  for (const sourceKind of ["empty", "example"]) {
    assert.equal(harness.view(nativeInventory([entry("backend.cuda")]), sourceKind).rows.length, 0);
  }
  assert.equal(harness.view({ device: { runtime: "cuda" }, accepted: true }).rows.length, 0);
});

test("native states and evidence tiers remain selected-document claims", () => {
  const harness = consoleHarness();
  const states = ["AVAILABLE", "UNAVAILABLE", "EXPERIMENTAL", "BUILD_DISABLED", "HARDWARE_UNSUPPORTED", "RUNTIME_UNVERIFIED"];
  const result = harness.view(nativeInventory(states.map((state, i) => entry("capability." + i, state))));
  assert.deepEqual(result.rows.map((row) => row.state), states);
  assert.ok(result.rows.every((row) => row.evidence.includes("claimed SOURCE_ONLY")));
  assert.ok(result.rows.every((row) => row.note.includes("does not independently verify")));
  assert.match(result.note, /build_inventory/);
  const oldEntry = entry("legacy");
  delete oldEntry.evidence_tier;
  assert.match(harness.view(nativeInventory([oldEntry])).rows[0].evidence, /tier unreported/);
});

test("the existing public catalog preserves disabled profiles without claiming execution", () => {
  const catalog = JSON.parse(fs.readFileSync(path.join(root, "configs/public/capabilities.json"), "utf8"));
  const result = consoleHarness().view(catalog);
  assert.deepEqual(result.rows.map((row) => [row.name, row.state]), [
    ["cuda-local", "cataloged"], ["opencl-smoke", "disabled"], ["cpu-smoke", "disabled"],
  ]);
  assert.ok(result.rows.every((row) => row.evidence === "catalog only"));
});

test("malformed, duplicate, and unknown inventory claims fail closed", () => {
  const harness = consoleHarness();
  for (const inventory of [
    nativeInventory(null),
    nativeInventory([entry("duplicate"), entry("duplicate")]),
    nativeInventory([entry("unknown", "PROMOTED")]),
    nativeInventory([{ ...entry("unknown-tier"), evidence_tier: "TRUST_ME" }]),
    nativeInventory([null]),
    { schema_version: "neural-foundry-capability-catalog.v1", profiles: [] },
    { schema_version: "neural-foundry-capability-catalog.v1", profiles: { local: { backend: "cuda", enabled: "false" } } },
  ]) {
    const result = harness.view(inventory);
    assert.equal(result.label, "INVALID INVENTORY");
    assert.equal(result.rows.length, 0);
  }
});

test("file selection refreshes the rendered table and routes to capabilities", async () => {
  const harness = consoleHarness();
  await harness.load(nativeInventory([entry("backend.cuda")]));
  assert.equal(harness.context.document.documentElement.dataset.view, "capabilities");
  assert.equal(harness.nodes.get("capability-rows").children[0].children[0].textContent, "backend.cuda");
  assert.equal(harness.nodes.get("capability-rows").children[0].children[1].textContent, "BUILD_DISABLED");
  await harness.load(nativeInventory([entry("backend.cpu", "UNAVAILABLE")]));
  assert.equal(harness.nodes.get("capability-rows").children.length, 1);
  assert.equal(harness.nodes.get("capability-rows").children[0].children[0].textContent, "backend.cpu");
  await harness.load(nativeInventory([entry("unknown", "PROMOTED")]));
  assert.equal(harness.nodes.get("capability-rows").children.length, 0);
  assert.equal(harness.nodes.get("capability-state-tag").textContent, "INVALID INVENTORY");
  await harness.load(nativeInventory([entry("backend.cuda")]));
  vm.runInContext("loadExample()", harness.context);
  assert.equal(harness.nodes.get("capability-rows").children.length, 0);
  assert.equal(harness.nodes.get("capability-state-tag").textContent, "NO INVENTORY");
});

test("inventory text is inserted as text rather than HTML", async () => {
  const harness = consoleHarness();
  const markup = '<img src=x onerror="throw 1">';
  await harness.load(nativeInventory([entry(markup)]));
  const name = harness.nodes.get("capability-rows").children[0].children[0];
  assert.equal(name.textContent, markup);
  assert.equal(name.innerHTML, undefined);
});
