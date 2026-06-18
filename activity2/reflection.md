# Activity 2 Reflection

## 1. Did the agent stay focused on the goal throughout the loop?
Yes. The `autonomous_planner` function repeatedly passes the same high-level objective (`goal`) into each model call, along with the current steps it has already decided. This keeps the model grounded on the same goal as it generates each sequential planning step.

## 2. Challenge: Modify the loop to perform 5 steps instead of 3. How does this affect the detail of the plan?
Using 5 steps instead of 3 generally increases the level of detail in the plan. The planner can break the objective into smaller, more specific actions and intermediate tasks, which makes the plan more granular and potentially easier to execute. However, it can also lead to more verbose output and may require stronger guidance to keep all steps relevant and non-redundant.

---

## Notes
- A `screenshots` folder was created in `activity2/screenshots` for saving output screenshots.
- The planner script is designed to print the final plan after executing the loop.
