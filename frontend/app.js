const appState = {
  view: "overview",
  document: null,
  documentName: "",
  documentType: "",
  source: "empty",
};

const exampleDocument = {
  schema: "canopy-foundry-execution-plan.v1",
  planner_revision: "planner-foundation-v2-port-v1",
  device: {
    runtime: "cuda",
    backend: "cuda",
    required_arch: "sm_61",
    parallel_mode: "single_device",
    transport: "none",
  },
  model: {
    architecture_contract: "ida_lattice_native_v1",
    family: "contract-fixture",
    layers: 4,
    hidden_size: 256,
    moe: false,
  },
  precision: {
    profile: "legacy_bf16",
    master_storage: "bf16",
    forward_execution: "bf16",
    backward_dx_execution: "bf16",
    gradients: "fp32",
    accumulators: "fp32",
    optimizer_state: "fp32",
  },
  optimizer: {
    type: "lion",
    state_precision: "fp32",
  },
  accepted: false,
  runtime_verification_required: true,
};

const capabilityRows = [
  { name: "backend.cuda", state: "cataloged", evidence: "public catalog", tone: "teal", note: "Build/profile surface; machine verification remains separate." },
  { name: "backend.opencl", state: "disabled", evidence: "public catalog", tone: "muted", note: "No silent fallback from a CUDA request." },
  { name: "precision.bf16", state: "contract", evidence: "planner-tested", tone: "teal", note: "Role mapping is explicit; execution still needs a runtime." },
  { name: "precision.nvfp4", state: "experimental", evidence: "contract surface", tone: "amber", note: "Presence of a name is not hardware evidence." },
  { name: "moe.route_plan", state: "contract", evidence: "unit-tested", tone: "teal", note: "Forward-owned route identity is available to the planner." },
  { name: "moe.compact_scheduler", state: "experimental", evidence: "design boundary", tone: "amber", note: "Not promoted by this frontend." },
  { name: "parallel.host_staged", state: "contract", evidence: "planner surface", tone: "teal", note: "Transport choice must be resolved explicitly." },
  { name: "checkpoint.generation", state: "contract", evidence: "unit-tested", tone: "teal", note: "Torn generations must fail closed." },
];

const evidenceStages = [
  ["Requested", "operator input", "current"],
  ["Resolved", "native planner", "contract"],
  ["Accepted", "admission gate", ""],
  ["Observed", "runtime counters", ""],
  ["Persisted", "durable receipt", ""],
];

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function text(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function valueAt(source, paths, fallback = "—") {
  for (const path of paths) {
    const value = path.split(".").reduce((current, key) => {
      if (current === null || current === undefined || typeof current !== "object") return undefined;
      return current[key];
    }, source);
    if (value !== undefined && value !== null && value !== "") {
      if (typeof value === "boolean") return value ? "true" : "false";
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }
  }
  return fallback;
}

function addFact(container, label, value, tone = "") {
  const row = document.createElement("div");
  row.className = "fact-row";
  const labelNode = document.createElement("span");
  labelNode.className = "fact-label";
  labelNode.textContent = label;
  const valueNode = document.createElement("span");
  valueNode.className = "fact-value" + (tone ? " tone-" + tone : "");
  valueNode.textContent = value;
  valueNode.title = value;
  row.append(labelNode, valueNode);
  container.append(row);
}

function renderFacts(id, facts) {
  const container = document.getElementById(id);
  if (!container) return;
  container.replaceChildren();
  facts.forEach((fact) => addFact(container, fact[0], fact[1], fact[2] || ""));
}

function classifyDocument(value) {
  const schema = valueAt(value, ["schema", "schema_version"], "").toLowerCase();
  if (schema.includes("receipt") || value.observed || value.persisted) return "execution receipt";
  if (schema.includes("capabilit") || Array.isArray(value.capabilities)) return "capability inventory";
  if (schema.includes("plan") || value.device || value.precision || value.required_capabilities) return "execution plan";
  return "local JSON document";
}

function currentDocument() {
  return appState.document || {};
}

function setPosture(title, copy, percent, tone = "amber") {
  text("evidence-posture-title", title);
  text("evidence-posture-copy", copy);
  text("rail-posture-title", title.toUpperCase());
  text("rail-posture-copy", copy);
  const postureMeter = $(".posture-meter span");
  const signalBar = $(".signal-bar span");
  if (postureMeter) postureMeter.style.width = String(percent) + "%";
  if (signalBar) signalBar.style.width = String(percent) + "%";
  [postureMeter, signalBar].forEach((node) => {
    if (!node) return;
    node.style.background = tone === "teal" ? "var(--teal)" : tone === "red" ? "var(--red)" : "var(--amber)";
  });
}

function renderDocument() {
  const doc = currentDocument();
  const type = appState.source === "empty" ? "no document" : appState.documentType;
  const sourceLabel = appState.source === "file" ? appState.documentName : appState.source === "example" ? "Example contract" : "No file";
  const stateLabel = appState.source === "empty" ? "NO DOCUMENT" : appState.source === "example" ? "EXAMPLE ONLY" : "LOCAL FILE";

  text("overview-source-tag", stateLabel);
  text("overview-source", sourceLabel);
  text("overview-document", type);
  text("plan-state-tag", stateLabel);
  text("plan-document-name", sourceLabel === "No file" ? "No local JSON loaded" : sourceLabel);
  text("rail-document-name", sourceLabel === "No file" ? "None loaded" : sourceLabel);
  text("rail-document-type", sourceLabel === "No file" ? "Waiting for local JSON" : type + " · browser memory only");
  text("source-note", appState.source === "file"
    ? appState.documentName + " loaded locally. No network request or runtime claim was made."
    : appState.source === "example"
      ? "Example contract loaded. Values are illustrative and not a runtime receipt."
      : "Nothing has been loaded. The example remains clearly marked as non-runtime data.");
  text("footer-message", appState.source === "file"
    ? "Loaded locally · " + appState.documentName + " · no upload performed."
    : appState.source === "example"
      ? "Example contract visible · no runtime evidence attached."
      : "Ready · select a local JSON document or inspect the example contract.");
  const json = document.getElementById("request-json");
  if (json) json.textContent = JSON.stringify(doc, null, 2);

  renderFacts("request-facts", [
    ["schema", valueAt(doc, ["schema", "schema_version"])],
    ["document type", type],
    ["backend", valueAt(doc, ["backend", "device.backend", "execution.backend"])],
    ["required arch", valueAt(doc, ["required_arch", "device.required_arch"])],
    ["model contract", valueAt(doc, ["model.architecture_contract", "model_contract_id"])],
    ["optimizer", valueAt(doc, ["optimizer_type", "optimizer.type", "optimizer"])],
  ]);

  renderFacts("plan-facts", [
    ["planner revision", valueAt(doc, ["planner_revision", "execution_plan.planner_revision"], "not present")],
    ["parallel mode", valueAt(doc, ["parallel_mode", "device.parallel_mode"], "single device")],
    ["transport", valueAt(doc, ["transport", "device.transport"], "none")],
    ["required capabilities", valueAt(doc, ["required_capabilities"], "not present")],
    ["accepted", valueAt(doc, ["accepted", "execution_plan.accepted"], "not present"), "amber"],
    ["runtime verification", valueAt(doc, ["runtime_verification_required", "execution_plan.runtime_verification_required"], "not present"), "amber"],
  ]);

  renderFacts("precision-facts", [
    ["master storage", valueAt(doc, ["precision.master_storage", "precision.master"])],
    ["forward execution", valueAt(doc, ["precision.forward_execution", "precision.forward"])],
    ["backward dX", valueAt(doc, ["precision.backward_dx_execution", "precision.backward_dx"])],
    ["gradients", valueAt(doc, ["precision.gradients", "precision.gradient"])],
    ["accumulators", valueAt(doc, ["precision.accumulators", "precision.accumulator"])],
    ["optimizer state", valueAt(doc, ["precision.optimizer_state", "optimizer.state_precision", "optimizer_state_precision"])],
  ]);

  const hasObserved = Boolean(doc.observed || doc.observation || doc.metrics);
  const hasPersisted = Boolean(doc.persisted || doc.checkpoint || doc.receipt_sha256);
  if (appState.source === "file" && (hasObserved || hasPersisted)) {
    setPosture(hasPersisted ? "Receipt attached" : "Observed fields", hasPersisted
      ? "A local receipt-shaped document is present. Verify its schema and integrity before treating it as durable evidence."
      : "Observed fields are present in the selected document; this console does not independently verify them.", hasPersisted ? 68 : 48, "amber");
  } else if (appState.source === "file") {
    setPosture("Request preview", "A local request or plan is loaded. Resolution and execution remain native-runtime responsibilities.", 28, "amber");
  } else if (appState.source === "example") {
    setPosture("Contract-only", "The example demonstrates a shape. It is intentionally not represented as a runtime result.", 24, "amber");
  } else {
    setPosture("Contract-only", "No runtime receipt is attached. The console is safe to explore, but it is not reporting an execution.", 24, "amber");
  }
}

function renderCapabilities() {
  const container = document.getElementById("capability-rows");
  if (!container) return;
  container.replaceChildren();
  capabilityRows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "capability-row";
    const name = document.createElement("span");
    name.className = "capability-name";
    name.textContent = item.name;
    const state = document.createElement("span");
    state.className = "capability-state" + (item.tone === "amber" ? " is-amber" : item.tone === "muted" ? " is-muted" : "");
    state.textContent = item.state;
    const evidence = document.createElement("span");
    evidence.className = "capability-evidence" + (item.tone === "amber" ? " is-amber" : item.tone === "muted" ? " is-muted" : "");
    evidence.textContent = item.evidence;
    const note = document.createElement("span");
    note.className = "capability-note";
    note.textContent = item.note;
    row.append(name, state, evidence, note);
    container.append(row);
  });
}

function renderEvidence() {
  const container = document.getElementById("evidence-track");
  if (!container) return;
  container.replaceChildren();
  evidenceStages.forEach(([name, detail, state]) => {
    const stage = document.createElement("div");
    stage.className = "evidence-stage";
    const node = document.createElement("div");
    node.className = "evidence-node" + (state === "current" ? " is-current" : state === "contract" ? " is-contract" : "");
    const title = document.createElement("strong");
    title.textContent = name;
    const caption = document.createElement("span");
    caption.textContent = detail;
    stage.append(node, title, caption);
    container.append(stage);
  });
  renderFacts("receipt-facts", [
    ["requested", "separate claim"],
    ["resolved", "native planner"],
    ["accepted", "pre-execution gate"],
    ["observed", "runtime counters"],
    ["persisted", "durable artifact"],
  ]);
}

function switchView(view) {
  if (!document.querySelector('[data-view-panel="' + view + '"]')) return;
  appState.view = view;
  document.documentElement.dataset.view = view;
  $$("[data-view]").forEach((button) => button.classList.toggle("is-active", button.dataset.view === view));
  $$("[data-view-panel]").forEach((panel) => {
    const active = panel.dataset.viewPanel === view;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  const label = view === "overview" ? "OVERVIEW" : view.toUpperCase();
  text("current-view-label", label);
  $(".main-content")?.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadSelectedFile() {
  const input = document.getElementById("request-file");
  const file = input?.files?.[0];
  if (!file) return;
  if (file.size > 2 * 1024 * 1024) {
    text("footer-message", "Rejected · local JSON is larger than the 2 MiB inspection limit.");
    input.value = "";
    return;
  }
  try {
    const parsed = JSON.parse(await file.text());
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("document must be a JSON object");
    appState.document = parsed;
    appState.documentName = file.name;
    appState.documentType = classifyDocument(parsed);
    appState.source = "file";
    renderDocument();
    switchView("plan");
  } catch (error) {
    text("footer-message", "Rejected · could not read " + file.name + ": " + error.message);
    input.value = "";
  }
}

function loadExample() {
  appState.document = exampleDocument;
  appState.documentName = "";
  appState.documentType = "execution plan";
  appState.source = "example";
  renderDocument();
  switchView("plan");
}

function updateClock() {
  const now = new Date();
  const clock = document.getElementById("clock");
  if (!clock) return;
  clock.dateTime = now.toISOString();
  clock.textContent = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(now);
}

function bindEvents() {
  $$("[data-view]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  $$("[data-view-jump]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.viewJump));
  });
  $$("[data-action='load-file']").forEach((button) => {
    button.addEventListener("click", () => document.getElementById("request-file")?.click());
  });
  $$("[data-action='example']").forEach((button) => button.addEventListener("click", loadExample));
  document.getElementById("request-file")?.addEventListener("change", loadSelectedFile);
}

bindEvents();
renderCapabilities();
renderEvidence();
renderDocument();
updateClock();
window.setInterval(updateClock, 1000);
