# Title     : Wikipedia dataset preprocessing
# Created by: pereramg
# Created on: 7/3/22

### Wikipedia page views (count)
# Daily data from 01-06-2016 to 29-06-2017
# Level 0: Total page views for wikipedia
# level 1: Language (de, en, es, zh)
# level 2: Language x Access (desktop, mobile app, mobile web)
# level 3: Language x Access x Agent (spider, user)
# level 4: Language x Access x Agent x Article (bottom level)

# install.packages('fable')

library(readr)
library(lubridate)
library(tidyverse)
library(fable)

wiki <- read_csv("data/wikipedia_data.csv")


wiki <- wiki %>%
  mutate(date = mdy(date)) %>%
  select(-project,-granularity) %>%
  rename(purpose = Purpose)

wiki <- wiki %>%
  group_by(date, language, access, agent, article) %>%
  summarise(total_views = sum(views))

# all the time series
aggts <- wiki %>%
  as_tsibble(key = c(language, access, article, agent)) %>%
  aggregate_key(language/access/agent/article, total_views = sum(total_views))

# bottom level
bts <- wiki %>%
  arrange(language, access, article, agent) %>% as_tibble() %>%
  unite(col = "Description", c(language:article), sep = "-") %>%
  mutate(Level = 5) %>% pivot_wider(names_from = date, values_from = total_views)

# aggregated by article
level3 <- aggts %>%
  filter(!is_aggregated(language), !is_aggregated(access),
         !is_aggregated(agent), is_aggregated(article)) %>%
  select(-article) %>%
  mutate(language = as.character(language), access = as.character(access),
         agent = as.character(agent)) %>%
  arrange(language, access, agent) %>% as_tibble() %>%
  unite(col = "Description", c(language:agent), sep = "-") %>%
  mutate(Level = 4) %>% pivot_wider(names_from = date, values_from = total_views)



# aggregated by article and agent
level2 <- aggts %>%
  filter(!is_aggregated(language), !is_aggregated(access),
         is_aggregated(agent), is_aggregated(article)) %>%
  select(-article, -agent) %>%
  mutate(language = as.character(language), access = as.character(access)) %>%
  arrange(language, access) %>% as_tibble() %>%
  unite(col = "Description", c(language:access), sep = "-") %>%
  mutate(Level = 3) %>% pivot_wider(names_from = date, values_from = total_views)


# aggregated by article, agent and access
level1 <- aggts %>%
  filter(!is_aggregated(language), is_aggregated(access),
         is_aggregated(access), is_aggregated(article)) %>%
  select(-article, -agent, -access) %>%
  as_tibble() %>%
  mutate(language = as.character(language)) %>%
  arrange(language) %>% mutate(Level = 2) %>%
  rename(Description = language) %>%
  pivot_wider(names_from = date, values_from = total_views)

# top level
top <- aggts %>%
  filter(is_aggregated(language), is_aggregated(access),
         is_aggregated(access), is_aggregated(article)) %>%
  select(-article, -agent, -access, -language) %>% as_tibble() %>%
  mutate(Level = 1, Description = "Aggregated") %>%
  pivot_wider(names_from = date, values_from = total_views)

all_level_ts <- rbind(top, level1, level2, level3, bts)
all_level_ts_test <- all_level_ts[, (ncol(all_level_ts) - 6):ncol(all_level_ts)]
all_level_ts_train <- all_level_ts[, 1: (ncol(all_level_ts) - 7)]


meta_info <- all_level_ts_train[, c(1,2)] # get level and description
final_test <- cbind(meta_info, all_level_ts_test)
write.table(final_test, "input_data/wikipedia_test.csv", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(all_level_ts_train, "input_data/wikipedia_actual.csv", col.names = TRUE, row.names = FALSE, sep = ",")