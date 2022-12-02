# Title     : Wikipedia dataset preprocessing
# Created by: pereramg
# Created on: 7/3/22

### Wikipedia page views (count)
# Daily data from 01-06-2016 to 29-06-2017
# Level 0: Total page views for wikipedia
# level 1: Access (desktop, mobile app, mobile web)
# level 2: Access x Agent (spider, user)
# level 3: Access X Agent X Language (de, en, es, zh)
# level 4: Access X Agent X Language x Purpose
# level 5: Access X Agent X Language x Purpose X Article

# install.packages('fable')

library(readr)
library(lubridate)
library(tidyverse)
library(fable)
library(forecast)

wiki <- read_csv("data/wikipedia_data.csv")


wiki <- wiki %>%
  mutate(date = mdy(date)) %>%
  select(-project,-granularity) %>%
  rename(purpose = Purpose)

wiki <- wiki %>%
  group_by(date, access, agent, language, purpose, article) %>%
  summarise(total_views = sum(views))

wiki %>%
  unite(col = "all_var", 2, 3, 4, 5, 6, sep = "-") %>%
  pivot_wider(names_from = date, values_from = total_views) -> wiki

wiki %>%
  split(.$all_var) %>%
  map(~ .x[-1]) %>%
  map_dfr(~ tsclean(ts(as.numeric(.x), frequency = 7))) %>%
  bind_cols(date = colnames(wiki)[-1], .) %>%
  mutate(date = as_date(date)) %>%
  pivot_longer(-1, names_to = "all_var", values_to = "total_views") %>%
  separate(col = 2,
           into = c("access", "agent", "language", "purpose", "article"),
           sep = "-") -> wiki

# all the time series
aggts <- wiki %>%
  as_tsibble(key = c(access, agent, language, purpose, article)) %>%
  aggregate_key(access/agent/language/purpose/article, total_views = sum(total_views))

# bottom level
bts <- wiki %>%
  arrange(access, agent, language, purpose, article) %>% as_tibble() %>%
  unite(col = "Description", c(access:article), sep = "-") %>%
  mutate(Level = 6) %>% pivot_wider(names_from = date, values_from = total_views)

# aggregated by article
level4 <- aggts %>%
  filter(!is_aggregated(access), !is_aggregated(agent),
         !is_aggregated(language), !is_aggregated(purpose), is_aggregated(article)) %>%
  select(-article) %>%
  mutate(access = as.character(access),
         agent = as.character(agent), language = as.character(language), purpose = as.character(purpose)) %>%
  arrange(access, agent, language, purpose) %>% as_tibble() %>%
  unite(col = "Description", c(access:purpose), sep = "-") %>%
  mutate(Level = 5) %>% pivot_wider(names_from = date, values_from = total_views)



# aggregated by article and purpose
level3 <- aggts %>%
  filter(!is_aggregated(access), !is_aggregated(agent),
         !is_aggregated(language), is_aggregated(purpose), is_aggregated(article)) %>%
  select(-article, -purpose) %>%
  mutate(access = as.character(access),
         agent = as.character(agent), language = as.character(language)) %>%
  arrange(access, agent, language) %>% as_tibble() %>%
  unite(col = "Description", c(access:language), sep = "-") %>%
  mutate(Level = 4) %>% pivot_wider(names_from = date, values_from = total_views)


# aggregated by article, purpose and language
level2 <- aggts %>%
  filter(!is_aggregated(access), !is_aggregated(agent),
         is_aggregated(language), is_aggregated(purpose), is_aggregated(article)) %>%
  select(-article, -purpose, -language) %>%
  as_tibble() %>%
  mutate(access = as.character(access),
         agent = as.character(agent)) %>%
  arrange(access, agent) %>% mutate(Level = 3) %>%
  unite(col = "Description", c(access:agent), sep = "-") %>%
  pivot_wider(names_from = date, values_from = total_views)


# aggregated by article, purpose and language and agent
level1 <- aggts %>%
  filter(!is_aggregated(access), is_aggregated(agent),
         is_aggregated(language), is_aggregated(purpose), is_aggregated(article)) %>%
  select(-article, -purpose, -language, -agent) %>%
  as_tibble() %>%
  mutate(access = as.character(access)) %>%
  arrange(access) %>% mutate(Level = 2) %>%
  rename(Description = access) %>%
  pivot_wider(names_from = date, values_from = total_views)

# top level
top <- aggts %>%
  filter(is_aggregated(access), is_aggregated(agent),
         is_aggregated(language), is_aggregated(purpose), is_aggregated(article)) %>%
  select(-article, -purpose, -language, -agent, -access) %>% as_tibble() %>%
  mutate(Level = 1, Description = "Aggregated") %>%
  pivot_wider(names_from = date, values_from = total_views)

all_level_ts <- rbind(top, level1, level2, level3, level4, bts)

write.table(all_level_ts, "input_data/wikipedia.csv", col.names = TRUE, row.names = FALSE, sep = ",")