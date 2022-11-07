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


labour %>%
  as_tsibble(key = c("Sex", "Employment_status", "Main_occupation")) %>%
  arrange(Sex, Employment_status, Main_occupation) %>%
  autoplot(Count) + theme(legend.position="none")

# ggsave('data/labour_original.png')

labour %>% unite(col = "all_var", 1,3,4, sep = "#") %>% pivot_wider(names_from = Month, values_from = Count) -> labour_out

labour_out %>%
  split(.$all_var) %>%
  map(~ .x[-1]) %>%
  map_dfr(~ tsclean(ts(as.numeric(.x), frequency = 4))) %>%
  bind_cols(Month = colnames(labour_out)[-1], .) %>%
  mutate(Month = yearmonth(Month)) %>%
  pivot_longer(-1, names_to = "all_var", values_to = "Count") %>%
  separate(col = 2,
           into = c("Sex", "Employment_status", "Main_occupation"),
           sep = "#") -> new_data_labour

new_data_labour %>%
  as_tsibble(key = c("Sex", "Employment_status", "Main_occupation")) %>%
  arrange(Sex, Employment_status, Main_occupation)  %>%
  autoplot(Count) + theme(legend.position="none")

write.table(new_data_labour, "input_data/labour_plot.csv", col.names = TRUE, row.names = TRUE, sep = ",")


ggsave('data/labour_clean.png')

