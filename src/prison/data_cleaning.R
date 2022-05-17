library(readr)
library(lubridate)
library(tidyverse)
library(fable)
library(tsibble)
library(forecast)

prison <- read_csv("data/prison_population.csv")
prison <- prison %>%
  mutate(Quarter = yearquarter(Date)) %>%
  select(-Date)

prison %>%
  as_tsibble(key = c("State", "Gender", "Legal", "Indigenous")) %>%
  arrange(State, Gender, Legal, Indigenous) %>%
  autoplot(Count) + theme(legend.position="none")

ggsave('data/prison_original.png')

prison %>% unite(col = "all_var", 1,2,3,4, sep = "#") %>% pivot_wider(names_from = Quarter, values_from = Count) -> prison_out

prison_out %>%
  split(.$all_var) %>%
  map(~ .x[-1]) %>%
  map_dfr(~ tsclean(ts(as.numeric(.x), frequency = 4))) %>%
  bind_cols(Quarter = colnames(prison_out)[-1], .) %>%
  mutate(Quarter = yearquarter(Quarter)) %>%
  pivot_longer(-1, names_to = "all_var", values_to = "Count") %>%
  separate(col = 2,
           into = c("State", "Gender", "Legal", "Indigenous"),
           sep = "#") -> new_data_prison
new_data_prison %>%
  as_tsibble(key = c("State", "Gender", "Legal", "Indigenous")) %>%
  arrange(State, Gender, Legal, Indigenous)  %>%
  autoplot(Count) + theme(legend.position="none")

ggsave('data/prison_clean.png')

