# Local Run must use the existing Hub-issued job/worker contract. An operator
# binary pin and process containment alone do not admit a training run.
option(CANOPY_FOUNDRY_ENABLE_LOCAL_OWNER
  "Reserved local owner prototype; unavailable without Hub-issued job admission"
  OFF)
if(CANOPY_FOUNDRY_ENABLE_LOCAL_OWNER)
  message(FATAL_ERROR
    "Local Run requires an existing Hub-issued job. The operator-only owner "
    "prototype has no Hub admission adapter and cannot be enabled. Use the "
    "existing Hub-managed worker contract.")
endif()
