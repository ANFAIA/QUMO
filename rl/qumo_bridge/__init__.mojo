"""Package qumo_bridge: the only place where qrl meets QUMO (the one package here allowed to import `qumo.*`).

* `circuits`    - a jobs-file circuit / target (qrl.json) to a QUMO `Circuit` / `Target`.
* `export_route` - router export to `qumo.rlroute.v0`: the parity episodes are played through QUMO's routing
                   environment (`qumo.compiler.ai_route.Episode`) and the vectors cross-checked with QUMO's runtime.
* `route_ppo`   - router PPO fine-tune (episodes in QUMO's routing environment).
* `eval_blocks` - block evaluation of a synthesis model against QUMO's exact / greedy synthesizers, the keep rule.

The drivers that need these (route_tool, export_route, train_route_ppo, eval_synth, eval_blocks) live under
`rl/drivers` and are built with `-I . -I rl` from the QUMO root.
"""
