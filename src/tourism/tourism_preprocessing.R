# Title     : Tourism Dataset preprocessing
# Created by: pereramg
# Created on: 5/2/22
# Level 0 - Australia (Aggregated)
# Level 1 - States (7 States)
# Level 2 - Regions (77 time series)

library(tidyverse)
library(readr)
library(tsibble)
library(lubridate)
library(fabletools)
library(forecast)

header <- read_csv("data/tourism.csv", n_max = 3, col_names = FALSE)

header <- header %>%
  t() %>%
  as_tibble() %>%
  fill(V1:V2, .direction = "down") %>%
  unite(name, V1, V2, V3, na.rm = TRUE, sep = "/") %>%
  pull()

tourism <- read_csv("data/tourism.csv", skip = 3, col_names = header)
tourism <- tourism %>%
  fill(Year, .direction = "down")
tourism <- tourism %>%
  pivot_longer(!Year:Month, names_to = "Variable", values_to = "Trips") %>%
  separate(Variable, c("State", "Region", "Purpose"), sep = "/")

tourism <- tourism %>%
  mutate(Year_month = yearmonth(ym(paste(Year, Month)))) %>%
  as_tsibble(key = c(State, Region, Purpose), index = Year_month) %>%
  select(Year_month, State:Trips)

tourism <- tourism %>%
  mutate(State = recode(State,
                        `New South Wales` = "NSW",
                        `Northern Territory` = "NT",
                        `Queensland` = "QLD",
                        `South Australia` = "SA",
                        `Tasmania` = "TAS",
                        `Victoria` = "VIC",
                        `Western Australia` = "WA"
  ))


bottom_level_ts <- tourism %>% group_by(State, Region) %>%
  summarise(Trips = sum(Trips)) %>%
  ungroup() %>% as_tibble() %>%
  unite(col = Region, 1,2, sep = "-", remove =TRUE) %>%
  pivot_wider(names_from = Year_month, values_from = Trips) %>%
  rename(Description = Region) %>% mutate(Level = 3) %>%
  relocate(Level)

second_level_ts <- tourism %>%
  group_by(State) %>%
  summarise(Trips = sum(Trips)) %>%
  ungroup() %>% as_tibble() %>%
  pivot_wider(names_from = Year_month, values_from = Trips) %>%
  rename(Description = State) %>% mutate(Level = 2) %>%
  relocate(Level)

top_ts <- tourism %>% as_tibble() %>%
  group_by(Year_month) %>%
  summarise(Trips = sum(Trips)) %>%
  ungroup() %>%
  pivot_wider(names_from = Year_month, values_from = Trips) %>%
  mutate(Level = 1, Description = "Aggregated") %>%
  relocate(Level, Description)


all_level_ts <- rbind(top_ts, second_level_ts, bottom_level_ts)
all_level_ts_test <- all_level_ts[, (ncol(all_level_ts) - 11):ncol(all_level_ts)]
all_level_ts_train <- all_level_ts[, 1: (ncol(all_level_ts) - 12)]


meta_info <- all_level_ts_train[, c(1,2)] # get level and description
final_test <- cbind(meta_info, all_level_ts_test)
write.table(final_test, "input_data/tourism_test.csv", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(all_level_ts_train, "input_data/tourism_actual.csv", col.names = TRUE, row.names = FALSE, sep = ",")