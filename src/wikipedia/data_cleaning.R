library(readr)
library(lubridate)
library(tidyverse)
library(fable)
library(forecast)

wiki <- read_csv("data/wikipedia_data.csv")

wiki <- wiki  %>%
  mutate(date = mdy(date)) %>%
  select(-project,-granularity) %>%
  rename(purpose = Purpose)


wiki <- wiki %>%
  group_by(date, access, agent, language, purpose, article) %>%
  summarise(total_views = sum(views))


wiki %>%
  as_tsibble(key = c("access", "agent", "language", "purpose", "article")) %>%
  arrange(access, agent, language, purpose, article) %>%
  autoplot(total_views) + theme(legend.position="none")

ggsave('data/wiki_original.png')

wiki %>%
  unite(col = "all_var", 2, 3, 4, 5, 6, sep = "-") %>%
  pivot_wider(names_from = date, values_from = total_views) -> out

out %>%
  split(.$all_var) %>%
  map(~ .x[-1]) %>%
  map_dfr(~ tsclean(ts(as.numeric(.x), frequency = 7))) %>%
  bind_cols(date = colnames(out)[-1], .) %>%
  mutate(date = as_date(date)) %>%
  pivot_longer(-1, names_to = "all_var", values_to = "total_views") %>%
  separate(col = 2,
           into = c("access", "agent", "language", "purpose", "article"),
           sep = "-") -> new_data
new_data %>%
  as_tsibble(key = c("access", "agent", "language", "purpose", "article")) %>%
  arrange(access, agent, language, purpose, article) %>%
  autoplot(total_views) + theme(legend.position="none")

ggsave('data/wiki_clean.png')