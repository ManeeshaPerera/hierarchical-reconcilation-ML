# Title     : Labour dataset preprocessing
# Created by: pereramg
# Created on: 7/3/22

### Australian labour Force (no. of people in thousands)
# Monthly data from August 1986 to June 2021 (Removing Covid Period and taking upto 2018)
# source: ABS website
# Level 0: Total employed individuals
# Level 1: Main occupation category
# Level 2: Main occupation x Employment status (Full time, Part-time)
# Level 3: Main occupation x Employment status x Gender (Female, Male) (bottom level)

library(readr)
library(lubridate)
library(tidyverse)
library(fable)
library(tsibble)
library(forecast)

labour <- read_csv("data/labour_data.csv")
month <- labour %>% pull(`Mid-quarter month`) %>% my()
labour <- labour %>%
  mutate(`Mid-quarter month` = yearmonth(month)) %>%
  rename(Month = `Mid-quarter month`,
         Sub_occupation = `Occupation sub-major group of main job: ANZSCO (2013) v1.2`) %>%
  pivot_longer(`Employed full-time ('000)`:`Employed part-time ('000)`,
               names_to = "Employment_status", values_to = "Count") %>%
  mutate(Employment_status = recode(Employment_status,
                                    `Employed full-time ('000)` = "Full time",
                                    `Employed part-time ('000)` = "Part time")) %>%
  mutate(`Main_occupation` = recode(Sub_occupation,
                                    "Managers nfd" = "Managers",
                                    "Chief Executives, General Managers and Legislators" = "Managers",
                                    "Farmers and Farm Managers" = "Managers",
                                    "Specialist Managers" = "Managers",
                                    "Hospitality, Retail and Service Managers" = "Managers",
                                    "Professionals nfd" = "Professionals",
                                    "Arts and Media Professionals" = "Professionals",
                                    "Business, Human Resource and Marketing Professionals" = "Professionals",
                                    "Design, Engineering, Science and Transport Professionals" = "Professionals",
                                    "Education Professionals" = "Professionals",
                                    "Health Professionals" = "Professionals",
                                    "ICT Professionals" = "Professionals",
                                    "Legal, Social and Welfare Professionals" = "Professionals",
                                    "Technicians and Trades Workers nfd" = "Technicians and Trades Workers",
                                    "Engineering, ICT and Science Technicians" = "Technicians and Trades Workers",
                                    "Automotive and Engineering Trades Workers" = "Technicians and Trades Workers",
                                    "Construction Trades Workers" = "Technicians and Trades Workers",
                                    "Electrotechnology and Telecommunications Trades Workers" = "Technicians and Trades Workers",
                                    "Food Trades Workers" = "Technicians and Trades Workers",
                                    "Skilled Animal and Horticultural Workers" = "Technicians and Trades Workers",
                                    "Other Technicians and Trades Workers" = "Technicians and Trades Workers",
                                    "Community and Personal Service Workers nfd" = "Community and Personal Service Workers",
                                    "Health and Welfare Support Workers" = "Community and Personal Service Workers",
                                    "Carers and Aides" = "Community and Personal Service Workers",
                                    "Hospitality Workers" = "Community and Personal Service Workers",
                                    "Protective Service Workers" = "Community and Personal Service Workers",
                                    "Sports and Personal Service Workers" = "Community and Personal Service Workers",
                                    "Clerical and Administrative Workers nfd" = "Clerical and Administrative Workers",
                                    "Office Managers and Program Administrators" = "Clerical and Administrative Workers",
                                    "Personal Assistants and Secretaries" = "Clerical and Administrative Workers",
                                    "General Clerical Workers" = "Clerical and Administrative Workers",
                                    "Inquiry Clerks and Receptionists" = "Clerical and Administrative Workers",
                                    "Numerical Clerks" = "Clerical and Administrative Workers",
                                    "Clerical and Office Support Workers" = "Sales Workers",
                                    "Other Clerical and Administrative Workers" = "Sales Workers",
                                    "Sales Workers nfd" = "Sales Workers",
                                    "Sales Representatives and Agents" = "Sales Workers",
                                    "Sales Assistants and Salespersons" = "Sales Workers",
                                    "Sales Support Workers" = "Sales Workers",
                                    "Machinery Operators and Drivers nfd" = "Machinery Operators and Drivers",
                                    "Machine and Stationary Plant Operators" = "Machinery Operators and Drivers",
                                    "Mobile Plant Operators" = "Machinery Operators and Drivers",
                                    "Road and Rail Drivers" = "Machinery Operators and Drivers",
                                    "Storepersons" = "Machinery Operators and Drivers",
                                    "Labourers nfd" = "Labourers",
                                    "Cleaners and Laundry Workers" = "Labourers",
                                    "Construction and Mining Labourers" = "Labourers",
                                    "Factory Process Workers" = "Labourers",
                                    "Farm, Forestry and Garden Workers" = "Labourers",
                                    "Food Preparation Assistants" = "Labourers",
                                    "Other Labourers" = "Labourers"
  ))

labour <- select(labour, -Sub_occupation)
labour <-  labour %>% group_by(Sex, Month, Employment_status, Main_occupation) %>% summarise(Count = sum(Count))


labour %>% unite(col = "all_var", 1,3,4, sep = "#") %>% pivot_wider(names_from = Month, values_from = Count) -> labour

labour %>%
  split(.$all_var) %>%
  map(~ .x[-1]) %>%
  map_dfr(~ tsclean(ts(as.numeric(.x), frequency = 4))) %>%
  bind_cols(Month = colnames(labour)[-1], .) %>%
  mutate(Month = yearmonth(Month)) %>%
  pivot_longer(-1, names_to = "all_var", values_to = "Count") %>%
  separate(col = 2,
           into = c("Sex", "Employment_status", "Main_occupation"),
           sep = "#") -> labour


aggts <- labour %>%
  as_tsibble(key = c(Sex, Main_occupation, Employment_status)) %>%
  aggregate_key(Main_occupation/Employment_status/Sex, Count = sum(Count))


bts <- labour %>%
  arrange(Main_occupation, Employment_status, Sex) %>% as_tibble() %>%
  unite(col = "Description", c(Main_occupation, Employment_status, Sex), sep = "-") %>%
  mutate(Level = 4) %>% pivot_wider(names_from = Month, values_from = Count)


# aggregated by sex
level2 <- aggts %>%
  filter(!is_aggregated(Main_occupation),
         !is_aggregated(Employment_status), is_aggregated(Sex)) %>%
  select(-Sex) %>%
  mutate(Main_occupation = as.character(Main_occupation),
         Employment_status = as.character(Employment_status)) %>%
  arrange(Main_occupation, Employment_status) %>% as_tibble() %>%
  unite(col = "Description", c(Main_occupation, Employment_status), sep = "-") %>%
  mutate(Level = 3) %>% pivot_wider(names_from = Month, values_from = Count)


# aggregated by sex and employment status
level1 <- aggts %>%
  filter(!is_aggregated(Main_occupation),
         is_aggregated(Employment_status), is_aggregated(Sex)) %>%
  select(-Employment_status, -Sex) %>%
  mutate(Main_occupation = as.character(Main_occupation)) %>%
  arrange(Main_occupation)%>% as_tibble() %>%
  rename(Description = Main_occupation) %>%
  mutate(Level = 2) %>% pivot_wider(names_from = Month, values_from = Count)

# top level
top <- aggts %>%
  filter(is_aggregated(Main_occupation),
         is_aggregated(Employment_status), is_aggregated(Sex)) %>%
  select(-Main_occupation, -Employment_status, -Sex) %>% as_tibble() %>%
  mutate(Level = 1, Description = "Aggregated") %>%
  pivot_wider(names_from = Month, values_from = Count)

all_level_ts <- rbind(top, level1, level2, bts)

all_level_ts <- all_level_ts[, c(1, 2, 5:(ncol(all_level_ts)-10))] # remove 2019
write.table(all_level_ts, "input_data/labour.csv", col.names = TRUE, row.names = FALSE, sep = ",")


