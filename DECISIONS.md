# DECISIONS.md

1. **Indexes.** `employees`: unique `emp_code`, so a code cannot be saved twice; `department + emp_code` for the department filter and trend; `joined_on` for the summary headcount. `attendance_logs`: unique `emp_code + date`, so one employee has one record per day (it also serves every lookup for one employee); `date + emp_code` for the list sort and the leaderboard month; `status + date + emp_code` for the status filter. I rejected an `emp_code + punch_in` index for punch-out, because the latest date is always the latest punch-in.

2. **Punch-in race.** Both requests get the same date and call `insert_one` directly. I do not check first and then insert. MongoDB's unique index saves only the first one. The second gets `DuplicateKeyError`, and I return 409.

3. **Ties.** I use `$rank`, which gives the same rank to equal totals and skips the next number (1, 2, 2, 4). I apply `limit` after ranking, so everyone tied at the cutoff is returned, sorted by `emp_code`.

4. **Headcount.** The pipeline starts from `employees`, not from the logs. `$unwind` keeps employees with no logs, so they are counted.

5. **100x data.** I would keep a small daily summary per department, updated on every write, so analytics do not scan all logs.
