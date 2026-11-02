# Canopy Foundry local execution console

This is a dependency-free, static inspection frontend for Canopy Foundry. It
uses the quiet, local-first interaction pattern from HAVEN while preserving
Foundry's product boundary:

- the browser reads a user-selected JSON document into memory;
- the surface can show request, plan, capability, and receipt-shaped documents;
- the native Foundry planner remains the authority for resolution;
- no network request, socket, worker launch, authentication, or promotion action
  is implemented here.

Open index.html directly in a browser, or serve this directory with any static
file server. The interface intentionally labels example data and local
documents as non-runtime evidence unless a bounded receipt-shaped document is
provided for review. It does not independently verify a receipt.

The Capabilities view reads the selected
`canopy-foundry-runtime-capabilities.v1` inventory or
`neural-foundry-capability-catalog.v1` catalog. It preserves declared disabled,
unavailable, experimental, and unverified states and labels evidence as a
document claim. Loading a request, example, or another file clears the previous
inventory. There is no built-in table of assumed machine capabilities.

The dependency-free behavioral checks for inventory selection, replacement,
legacy catalogs, and malformed claims run with:

    node --test tests/frontend_capabilities.test.cjs

The intended next integration is a local adapter that supplies the output of
the native foundry plan and foundry receipt commands through files or an
explicit embedding surface. The adapter should not turn this directory into an
HTTP authority layer.
