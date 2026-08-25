# Optional Hugging Face/Git local cache

Canopy Foundry does not require Hugging Face, Git-over-HTTPS, Python model
libraries, or network access to build or run the native trainer. This cache is
an optional convenience for users who keep their own model repositories and
artifacts locally.

The cache root is `.local-cache/hf-git/`. Its contents are ignored by Git;
only the pointer README is tracked. Do not put tokens in source files, URLs,
shell history, or committed configuration.

## Enable it for a PowerShell session

From the repository root, dot-source the activation script so its environment
changes remain in the current shell:

```powershell
. .\scripts\enable_hf_git_cache.ps1
```

This sets:

- `HF_HOME` to `.local-cache/hf-git/huggingface`;
- `HF_HUB_CACHE` to `.local-cache/hf-git/huggingface/hub`;
- `GIT_CONFIG_GLOBAL` to `.local-cache/hf-git/gitconfig`.

The script creates only empty cache/config directories. It does not log in,
download a model, upload a repository, or contact a remote service.

## Hugging Face authentication

With the cache enabled, use the current `hf` CLI interactively:

```powershell
hf auth login
hf auth whoami
hf cache list --cache-dir $env:HF_HUB_CACHE
```

Do not pass a token on the command line and do not use
`--add-to-git-credential` when Git Credential Manager is configured. The
interactive login stores the Hugging Face token under the selected local
`HF_HOME`; the token file is ignored and must remain local.

If the CLI is not installed, install or configure it separately. Canopy
Foundry does not add an HF or PyTorch dependency.

## Git credentials

To use HTTPS Git with the operating system credential manager, dot-source the
script with its optional setup switch:

```powershell
. .\scripts\enable_hf_git_cache.ps1 -ConfigureGitCredentialManager
```

This resets inherited credential helpers in the ignored repo-local Git config,
selects Git Credential Manager, and sets `credential.useHttpPath=true` so
credentials are scoped by host/path. A plaintext `store` helper is not allowed
to remain active through a lower-priority system configuration; the script
fails closed if any unexpected helper is still effective.
The secret is stored by Git Credential Manager in the operating system's
credential store, not in the repository.

Use SSH instead if Git Credential Manager is unavailable. Do not enable
`credential.helper store`, do not create a plaintext `.git-credentials` file
under this directory, and never put a token in an HTTPS URL.

Example after authentication:

```powershell
git clone https://huggingface.co/<namespace>/<model-repository> `
  .\.local-cache\hf-git\repos\<local-model-name>
```

The destination is inside the ignored cache root and remains machine-local.
Review the destination and repository license before using any artifact for
training.

## Inspect, prune, and disable

Check that representative token and cache paths are ignored:

```powershell
git check-ignore -v .local-cache/hf-git/huggingface/token
git check-ignore -v .local-cache/hf-git/huggingface/hub/example
```

Use the HF cache tools without moving files into the repository index:

```powershell
hf cache prune --cache-dir $env:HF_HUB_CACHE --dry-run
hf cache prune --cache-dir $env:HF_HUB_CACHE --yes
```

To disable the wiring, close the shell or remove the session variables:

```powershell
Remove-Item Env:HF_HOME, Env:HF_HUB_CACHE, Env:GIT_CONFIG_GLOBAL -ErrorAction SilentlyContinue
```

The native CLI only points to this guide. It never reads the cache or sends
cache paths, tokens, model bytes, or Git credentials to the Hub. The public
boundary validator permits ignored Git metadata only under this cache root;
nested repositories elsewhere remain a release failure.
