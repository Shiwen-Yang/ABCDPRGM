library(tidyverse)
Oracle_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/oracle_performance.csv")
RGD_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/RGD_performance.csv")

n = nrow(Oracle_performance)

df_performance = rbind(Oracle_performance, RGD_performance)
df_performance <- df_performance %>%
  mutate("method" = c(rep("Ora_Align", n), rep("RGD_Align", n))) %>%
  as_tibble() %>%
  pivot_longer(cols = contains("error_"),
               names_to = "time",
               values_to = "error") %>%
  mutate(time = as.factor(str_extract(time, "\\d+")))


df_performance %>% ggplot() +
  geom_point(aes(x = nodes, y = error, color = time)) + 
  facet_wrap(~method, scale = "free")


