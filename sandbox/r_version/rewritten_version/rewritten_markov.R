library(tidyverse)
library(ranger)
library(expm)
library(foreach)
library(doParallel)
## Luuk replication

training_ids <- 121:408
test_ids <- 409:456
## Load data

# OLD DATA
#cm <- arrow::read_parquet('data/testdataset.parquet') %>%
#  select(-gleditsch_ward, -ln_ged_sb_dep, -ged_sb)

# NEW DATA
cm <- arrow::read_parquet('../../data/david_data.parquet') %>%
  select(-gleditsch_ward)


markov_fatalities_run <- function(
  cm,
  steps,
  training_ids,
  test_ids,
  method = c('direct', 'transition_matrix')
) {
  cm <- cm %>%
    group_by(country_id) %>%
    mutate(
      target_sb = lead(ln_ged_sb, steps),
      target_month_id = lead(month_id, steps),
      markov_state = case_when(
        ln_ged_sb > 0 & lag(ln_ged_sb, 1) > 0 ~ 'conflict',
        ln_ged_sb == 0 & lag(ln_ged_sb, 1) == 0 ~ 'peace',
        ln_ged_sb > 0 & lag(ln_ged_sb, 1) == 0 ~ 'escalation',
        ln_ged_sb == 0 & lag(ln_ged_sb, 1) > 0 ~ 'deescalation',
        TRUE ~ NA_character_
      ),
      target_state = lead(markov_state, steps),
      markov_state = factor(
        markov_state,
        levels = c('peace', 'escalation', 'conflict', 'deescalation')
      ),
      target_state = factor(
        target_state,
        levels = c('peace', 'escalation', 'conflict', 'deescalation')
      )
    ) %>%
    ungroup() %>%
    na.omit()

  if (method == 'direct') {
    cm <- cm %>%
      mutate(
        target_state_rf = target_state,
        target_sb_rf = target_sb
      )
  } else {
    cm <- cm %>%
      group_by(country_id) %>%
      mutate(
        target_sb_rf = lead(ln_ged_sb, 1),
        target_state_rf = lead(markov_state, 1),
        target_state_rf = factor(
          target_state_rf,
          levels = c('peace', 'escalation', 'conflict', 'deescalation')
        )
      ) %>%
      ungroup() %>%
      na.omit()
  }

  cm_train <- cm %>%
    filter(target_month_id %in% training_ids)
  cm_test <- cm %>%
    filter(target_month_id %in% test_ids)

  rf_regression_models <- cm_train %>%
    filter(!(target_state_rf %in% c('peace', 'deescalation'))) %>%
    group_by(target_state_rf) %>%
    group_split() %>%
    map(
      ~ list(
        model = ranger(
          target_sb_rf ~ .,
          data = .x %>%
            select(
              -country_id,
              -month_id,
              -target_month_id,
              -target_state,
              -target_sb,
              -markov_state,
              -target_state_rf
            )
        ),
        state = unique(.x$target_state)
      )
    )

  rf_point_predictions <- foreach(m = 1:2, .final = bind_cols) %do%
    {
      pp <- predict(
        rf_regression_models[[m]]$model,
        data = cm_test %>%
          select(
            -country_id,
            -month_id,
            -target_month_id,
            -target_state,
            -target_sb,
            -markov_state,
            -target_state_rf,
            -target_sb_rf
          )
      )$predictions
      
      tibble(!!paste0('pp_', rf_regression_models[[m]]$state) := pp)
    }

  rf_transition_models <- cm_train %>%
    group_by(markov_state) %>%
    group_split() %>%
    map(
      ~ list(
        model = ranger(
          target_state_rf ~ .,
          data = .x %>%
            select(
              -country_id,
              -month_id,
              -target_month_id,
              -target_sb,
              -target_sb_rf,
              -markov_state,
              -target_state,
            ),
          probability = TRUE
        ),
        state = unique(.x$markov_state)
      )
    )

  state_probs <- foreach(m = 1:4) %do%
    {
      pr <- predict(
        rf_transition_models[[m]]$model,
        data = cm_test %>%
          select(
            -country_id,
            -month_id,
            -target_month_id,
            -target_sb,
            -target_sb_rf,
            -markov_state,
            -target_state,
            -target_state_rf
          ),
      )$predictions
      pr <- as_tibble(pr)
      if (!'peace' %in% colnames(pr)) {
        pr <- pr %>%
          mutate(peace = 0)
      }
      if (!'deescalation' %in% colnames(pr)) {
        pr <- pr %>%
          mutate(deescalation = 0)
      }
      if (!'conflict' %in% colnames(pr)) {
        pr <- pr %>%
          mutate(conflict = 0)
      }
      if (!'escalation' %in% colnames(pr)) {
        pr <- pr %>%
          mutate(escalation = 0)
      }
      pr <- pr %>%
        select(peace, deescalation, conflict, escalation)
      colnames(pr) <- paste0('pr_', colnames(pr))
      pr <- pr %>%
        mutate(origin_state = rf_transition_models[[m]]$state, .before = 1)
      pr
    }

  transition_probabilities <- state_probs_to_transition_probabilities(
    state_probs,
    cm_test,
    steps = steps,
    method = method
  )

  results <- bind_cols(
    cm_test %>%
      select(country_id, month_id, markov_state, target_sb, target_state),
    rf_point_predictions,
    transition_probabilities
  ) %>%
    mutate(
      markov_point_prediction = pp_escalation *
        pr_escalation +
        pp_conflict * pr_conflict,
      method = method,
      steps = steps
    )

  return(results)
}

state_probs_to_transition_probabilities <- function(
  state_probs,
  cm_test,
  steps = NULL,
  method = c('direct', 'transition_matrix')
) {
  states <- c('peace', 'deescalation', 'escalation', 'conflict')
  n_states <- length(states)
  n_obs <- nrow(state_probs[[1]])

  transition_probabilities <- foreach(i = 1:n_obs, .final = bind_rows) %do%
    {
      tm <- state_probs %>%
        map(~ slice(.x, i)) %>%
        bind_rows() %>%
        arrange(match(origin_state, states)) %>%
        select(
          pr_peace,
          pr_deescalation,
          pr_escalation,
          pr_conflict
        ) %>%
        as.matrix()
      colnames(tm) <- paste0('pr_', states)
      rownames(tm) <- states
      current_state <- cm_test %>%
        slice(i) %>%
        pull(markov_state)
      tm_current <- tm[current_state, ]
      if (method == 'transition_matrix' & !is.null(steps)) {
        tm_current <- tm_current %*% (tm %^% (steps - 1))
        return(as_tibble(tm_current))
      }
      return(as_tibble_row(tm_current))
    }
  return(transition_probabilities)
}


registerDoParallel(cores = 7)

tictoc::tic()
test1 <- foreach(s = 1:2) %:%
  foreach(m = c('direct', 'transition_matrix')) %dopar%
  {
    cat('Running steps =', s, 'method =', m, '\n')
    markov_fatalities_run(
      cm,
      steps = s,
      training_ids,
      test_ids,
      method = m
    )
  }
tictoc::toc()

saveRDS(test1, 'results/markov_fatalities_views_test.rds')
