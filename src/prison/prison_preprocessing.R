### Australian Prison data (count)
# Actually a grouped structure but assumed to be hierarchical
# Quarterly data from 2005-2016
# source: fpp3 book
# Level 0: Australia
# level 1: State (ACT, NSW, NT, QLD, SA, TAS, VIC, WA)
# level 2: State x Gender (Female, Male)
# level 3: State x Gender x Legal (Remanded, Sentenced)
# Level 4: State x Gender x Legal x Indigenous (ATSI[Aboriginal and Torres Strait Islander], Non-ATSI) (bottom level)


# install.packages('readr')
# install.packages('lubridate')
# install.packages('tidyverse')
# install.packages('fable')

library(readr)
library(lubridate)
library(tidyverse)
library(fable)
library(tsibble)
library(forecast)

set.seed(1234)

prison <- read_csv("data/prison_population.csv")
prison <- prison %>%
  mutate(Quarter = yearquarter(Date)) %>%
  select(-Date)
aggts <- prison %>%
  as_tsibble(key = c(State, Gender, Legal, Indigenous)) %>%
  aggregate_key(State/Gender/Legal/Indigenous, Count = sum(Count))

# bottom level
bts <- prison %>%
  as_tsibble(key = c(State, Gender, Legal, Indigenous)) %>%
  arrange(State, Gender, Legal, Indigenous) %>% as_tibble() %>%
  unite(col = "Description", c(State:Indigenous), sep = "-") %>%
  mutate(Level = 5) %>% pivot_wider(names_from = Quarter, values_from = Count)

# aggregated by indigenous
level3 <- aggts %>%
  filter(!is_aggregated(State), !is_aggregated(Gender),
         !is_aggregated(Legal), is_aggregated(Indigenous)) %>%
  select(-Indigenous) %>%
  mutate(State = as.character(State), Gender = as.character(Gender),
         Legal = as.character(Legal)) %>%
  arrange(State, Gender, Legal) %>% as_tibble() %>%
  unite(col = "Description", c(State:Legal), sep = "-") %>%
  mutate(Level = 4) %>% pivot_wider(names_from = Quarter, values_from = Count)

# aggregated by legal status and indigenous
level2 <- aggts %>%
  filter(!is_aggregated(State), !is_aggregated(Gender),
         is_aggregated(Legal), is_aggregated(Indigenous)) %>%
  select(-Legal, -Indigenous) %>%
  mutate(State = as.character(State), Gender = as.character(Gender)) %>%
  arrange(State, Gender) %>% as_tibble() %>%
  unite(col = "Description", c(State:Gender), sep = "-") %>%
  mutate(Level = 3) %>% pivot_wider(names_from = Quarter, values_from = Count)


# aggregated by gender, legal status and indigenous
level1 <- aggts %>%
  filter(!is_aggregated(State), is_aggregated(Gender),
         is_aggregated(Legal), is_aggregated(Indigenous)) %>%
  select(-Gender, -Legal, -Indigenous) %>%
  as_tibble() %>%
  mutate(State = as.character(State)) %>%
  arrange(State) %>% mutate(Level = 2) %>%
  rename(Description = State) %>%
  pivot_wider(names_from = Quarter, values_from = Count)

# top level
top <- aggts %>%
  filter(is_aggregated(State), is_aggregated(Gender),
         is_aggregated(Legal), is_aggregated(Indigenous)) %>%
  select(-State, -Gender, -Legal, -Indigenous) %>% as_tibble() %>%
  mutate(Level = 1, Description = "Aggregated") %>%
  pivot_wider(names_from = Quarter, values_from = Count)

all_level_ts <- rbind(top, level1, level2, level3, bts)


all_level_ts_test <- all_level_ts[, (ncol(all_level_ts) - 7):ncol(all_level_ts)]
all_level_ts_train <- all_level_ts[, 1: (ncol(all_level_ts) - 8)]

meta_info <- all_level_ts_train[, c(1,2)]

final_test <- cbind(meta_info, all_level_ts_test)
write.table(final_test, "input_data/prison_test.csv", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(all_level_ts_train, "input_data/prison_actual.csv", col.names = TRUE, row.names = FALSE, sep = ",")