# Daily Standup Tracker

Use this file for ongoing daily standups for the rest of the project.

Keep one row per working day in the same table.

| Date | Yesterday | Today | Blockers |
| --- | --- | --- | --- |
| 2026-06-05 | Completed and verified the minimal two-tower retrieval pipeline. Ran filtered full ID-only two-tower training and evaluation. Confirmed that ID-only two-tower is technically working but still much weaker than the weighted popularity baseline. Added a history-aware query tower and verified that history features improve over ID-only. Added event-weighted history pooling and compared it against unweighted history. Ran metadata feasibility analysis for RetailRocket item metadata and category features. Confirmed that item category metadata is usable and well-covered enough for modeling. Implemented and evaluated a category-aware history two-tower with category embedding on the item tower. Found that category embedding gives some validation lift but is not a clear improvement over history-only on the capped test slice. | Finalize concise mentor-reviewable experiment summaries and standup context. Use the completed experiment results to define the next modeling gate. Prioritize richer session/history construction or deeper item/category feature modeling rather than more ID-only training. Keep comparisons anchored to the weighted popularity baseline, ID-only two-tower, unweighted history tower, event-weighted history tower, and category-aware history tower. | No hard blocker. Main caution is CPU runtime; filtered training and capped evaluation can still take several minutes per gate. Current learned two-tower variants remain below the weighted popularity baseline, so the next step should focus on representation quality rather than FAISS, serving, or MLOps. |
| YYYY-MM-DD |  |  |  |
