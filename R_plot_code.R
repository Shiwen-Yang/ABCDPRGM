library(tidyverse)
library(ggtern)
Oracle_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/oracle_performance.csv")
RGD_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/RGD_performance.csv")

n = nrow(Oracle_performance)
df_performance = rbind(Oracle_performance, RGD_performance)
df_performance <- df_performance %>%
  mutate("method" = c(rep("Ora_Align", n), rep("RGD_Align", n))) %>%
  as_tibble() %>%
  pivot_longer(cols = contains("error_"),
               names_to = "Time",
               values_to = "Error") %>%
  mutate(time = as.factor(str_extract(Time, "\\d+")))
df_performance %>% ggplot() +
  geom_point(aes(x = nodes, y = Error, color = Time)) + 
  facet_wrap(~method, scale = "free")


path_ASE_aligned <- "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/estimate_lat_pos/ASE_aligned_lat_pos.csv"
path_ASE <- "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/estimate_lat_pos/ASE_lat_pos.csv"
path_RGD_aligned <- "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/estimate_lat_pos/RGD_aligned_lat_pos.csv"
path_truth <- "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/estimate_lat_pos/tru_lat_pos.csv"
ASE_aligned <- read.csv(path_ASE_aligned) %>% mutate(method = "ASE_aligned")
ASE <- read.csv(path_ASE) %>% mutate(method = "ASE")
RGD_aligned <- read.csv(path_RGD_aligned) %>% mutate(method = "RGD_aligned")
truth <- read.csv(path_truth) %>% mutate(method = "Truth")

all_lat_pos <- rbind(truth, ASE, ASE_aligned, RGD_aligned)[,-1] %>% 
  as_tibble() %>%
  mutate(group = as.factor(group),
         time = as.factor(time)) 

all_lat_pos %>% 
  filter(time == 0) %>%
  ggplot()+
  geom_point(aes(x = dim_1, y = dim_2, color = group, size = 0.1)) +
  scale_size_continuous(range = c(0.1, 0.2)) +
  facet_wrap(~method, scale = "free")

all_lat_pos %>% 
  filter(time == 1) %>%
  ggplot()+
  geom_point(aes(x = dim_1, y = dim_2, color = group, size = 0.1)) +
  scale_size_continuous(range = c(0.1, 0.2)) +
  facet_wrap(~method, scale = "free")
  


