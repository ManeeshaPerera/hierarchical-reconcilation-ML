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

# tourism %>%
#   autoplot(Trips) + theme(legend.position="none")
#
# ggsave('data/tourism_original.png')


tourism %>% unite(col = "all_var", 2,3,4, sep = "#") %>% pivot_wider(names_from = Year_month, values_from = Trips) -> tourism_out

tourism_out %>%
  split(.$all_var) %>%
  map(~ .x[-1]) %>%
  map_dfr(~ tsclean(ts(as.numeric(.x), frequency = 12))) %>%
  bind_cols(Year_month = colnames(tourism_out)[-1], .) %>%
  mutate(Year_month = yearmonth(Year_month)) %>%
  pivot_longer(-1, names_to = "all_var", values_to = "Trips") %>%
  separate(col = 2,
           into = c("State", "Region", "Purpose"),
           sep = "#") -> new_data_tourism

# print(tourism)
# print(new_data_tourism)

new_data_tourism %>% as_tsibble(key = c("State", "Region", "Purpose")) %>%
  arrange(State, Region, Purpose) %>%autoplot(Trips) + theme(legend.position="none")

ggsave('data/tourism_clean.png')
