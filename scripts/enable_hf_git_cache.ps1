[CmdletBinding()]
param(
    [switch]$ConfigureGitCredentialManager
)

$foundryRoot = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $foundryRoot ".local-cache\hf-git"
$hfHome = Join-Path $cacheRoot "huggingface"
$hfHubCache = Join-Path $hfHome "hub"
$gitConfig = Join-Path $cacheRoot "gitconfig"

New-Item -ItemType Directory -Force -Path $cacheRoot, $hfHome, $hfHubCache | Out-Null

$env:HF_HOME = $hfHome
$env:HF_HUB_CACHE = $hfHubCache
$env:GIT_CONFIG_GLOBAL = $gitConfig

if ($ConfigureGitCredentialManager) {
    $manager = Get-Command git-credential-manager -ErrorAction SilentlyContinue
    if (-not $manager) {
        $manager = Get-Command git-credential-manager-core -ErrorAction SilentlyContinue
    }
    if (-not $manager) {
        throw "Git Credential Manager was not found. Use SSH or install Git Credential Manager; do not use plaintext credential.helper store."
    }
    # An empty helper resets lower-priority system helpers (including a
    # possible plaintext "store") for this repo-local Git configuration.
    & git config --global --replace-all credential.helper ""
    if ($LASTEXITCODE -ne 0) {
        throw "Could not reset inherited Git credential helpers in the ignored repo-local configuration."
    }
    & git config --global --add credential.helper manager
    if ($LASTEXITCODE -ne 0) {
        throw "Could not write the ignored repo-local Git credential configuration."
    }
    & git config --global --replace-all credential.useHttpPath true
    if ($LASTEXITCODE -ne 0) {
        throw "Could not enable path-scoped Git credentials."
    }
    $effectiveHelpers = @(& git config --get-all credential.helper 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not verify the effective Git credential helper chain."
    }
    $unexpectedHelpers = @($effectiveHelpers | Where-Object { $_ -and $_ -ne "manager" })
    if ($unexpectedHelpers.Count -gt 0) {
        throw "An unexpected Git credential helper remains active; use SSH or remove it before authenticating."
    }
}

[pscustomobject]@{
    CacheRoot = $cacheRoot
    HFHome = $env:HF_HOME
    HFHubCache = $env:HF_HUB_CACHE
    GitConfigGlobal = $env:GIT_CONFIG_GLOBAL
    CredentialManagerConfigured = [bool]$ConfigureGitCredentialManager
}
