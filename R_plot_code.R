library(tidyverse)
library(cowplot)
Oracle_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/oracle_performance.csv")
RGD_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/RGD_performance.csv")

# Comparing Alignment: Oracle vs. RGD -------------------------------------

n = nrow(Oracle_performance)
df_performance = rbind(Oracle_performance, RGD_performance)
df_performance <- df_performance %>%
  mutate("method" = c(rep("Ora_Align", n), rep("RGD_Align", n))) %>%
  as_tibble() %>%
  pivot_longer(cols = contains("error_"),
               names_to = "Time",
               names_prefix = "error_",
               values_to = "Error") %>%
  mutate(Time = if_else(
    Time == "T0", "T = 0", "T = 1"
  ))

df_performance %>%
  ggplot(aes(x = nodes, y = Error, color = Time, shape = method)) +
  geom_point(size = 6) +
  labs(
    title = "Align: Oracle vs. RGD -- Example 1",
    x = "Nodes",
    y = "Error",
    color = "Time",
    shape = "Method Used to Align"
  ) +
  theme(
    text = element_text(size = 16, color = "black"),          # Default text size and color
    axis.title.x = element_text(size = 20, face = "bold", margin = margin(t = 30, b = 30, unit = "pt")), 
    axis.title.y = element_text(size = 20, face = "bold", margin = margin(l = 30, r = 30, unit = "pt")), # x and y axis labels
    axis.text = element_text(size = 16),                      # x and y axis tick labels
    legend.title = element_text(size = 14, face = "bold"),    # Legend title
    legend.text = element_text(size = 14),                    # Legend text
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5, margin = margin(t= 30, b = 30, unit = "pt"))  # Plot title
  ) +
  scale_shape_manual(values = c(3,4))




# Comparing Alignment Method ----------------------------------------------

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
  geom_point(aes(x = dim_1, y = dim_2, color = group), size = 0.1) +
  facet_wrap(~method, scale = "free")

all_lat_pos %>% 
  filter(time == 1) %>%
  ggplot()+
  geom_point(aes(x = dim_1, y = dim_2, color = group), size = 0.1) +
  facet_wrap(~method, scale = "free")
  


# Estimator Consistency and Variance --------------------------------------
# Accuracy 

path_consistency_linreg = "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/theo_var_vs_emp_var/justify_oracle_guess/lin_reg_init_est.csv"
path_consistency_oracle = "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/theo_var_vs_emp_var/justify_oracle_guess/oracle_init_est.csv"
consistency_linreg <- read.csv(path_consistency_linreg) %>% mutate(init = "linreg")
consistency_oracle <- read.csv(path_consistency_oracle) %>% mutate(init = "oracle")

constraint <- c(1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -0., -0.,  0.,  0.,  0.,
                0.,  0.,  1.,  0.,  0.,  0., -1., -0., -0.,  0.,  0.,  1.,  0.,  0.,
                0.,  0.,  0.,  0., -0., -1., -0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
                0.,  0., -0., -1., -0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                -0., -0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -0., -0.,
                -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.) %>% matrix(4, 21)
H <- solve(tcrossprod(constraint), constraint)

consistency_both <- bind_rows(consistency_linreg, consistency_oracle) %>% 
  pivot_longer(cols = starts_with("method_"),
               names_to = "method",
               names_prefix = "method_",
               values_to = "value") %>%
  filter(value == 1) %>%
  select(-value)

comparison_summary <- consistency_both %>% 
  group_by(nodes, seed, init, method) %>%
  reframe(C_B_est = c(H %*% B_est),
          C_B_real = c(H %*% B_real),
          component = 1:4,
          info_lost = mean(info_lost),
          time_elapsed = mean(time_elapsed),
          max_iterations = mean(max_iterations)) %>%
  group_by(nodes, init, method, component) %>%
  mutate(error = (C_B_est - C_B_real)) %>%
  summarize(mean_C_B_est = mean(C_B_est),
            sd_C_B_est = sd(C_B_est),
            mean_C_B_real = mean(C_B_real),
            bias = mean(error),
            sd_error = sd(error),
            mean_info_lost = mean(info_lost), 
            mean_time_elapsed = mean(time_elapsed),
            sd_time_elapsed = sd(time_elapsed))

comparison_summary %>% 
  ggplot(aes(x = nodes, y = bias, ymin = bias - 2 * sd_error, ymax = bias + 2 * sd_error, color = method, shape = init)) +
  geom_pointrange(size = 1, position = position_dodge(width =  450)) +
  facet_wrap(~component, scale = "free")

comparison_summary %>%
  ggplot(aes(x = nodes, y = mean_time_elapsed, ymin = mean_time_elapsed - 2 * sd_time_elapsed, ymax = mean_time_elapsed + 2 * sd_time_elapsed,color = method, shape = init)) + 
  geom_pointrange(position = position_dodge(width = 200), size = 1)


# Empirical Variance vs. Theoretical Variance -----------------------------

path_B_est <- "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/theo_var_vs_emp_var/B_oracle.csv"
B_est <- read.csv(path_B_est)
constraint <- c(1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -0., -0.,  0.,  0.,  0.,
                0.,  0.,  1.,  0.,  0.,  0., -1., -0., -0.,  0.,  0.,  1.,  0.,  0.,
                0.,  0.,  0.,  0., -0., -1., -0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
                0.,  0., -0., -1., -0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                -0., -0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -0., -0.,
                -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.) %>% matrix(4, 21)

  
B_est %>% 
  select(-X) %>%
  pivot_longer(cols = starts_with("method_"),
               values_to = "value",
               names_to = "method",
               names_prefix = "method_") %>%
  filter(value == 1) %>%
  group_by(seed, nodes, method) %>%
  reframe(C_B_est = c(solve(tcrossprod(constraint), constraint %*% B_est)),
          C_B_real = c(solve(tcrossprod(constraint), constraint %*% B_real)),
          component = paste0("component_", 1:4)) %>%
  mutate(abs_error = abs(C_B_est - C_B_real)) %>%
  group_by(nodes, method, component) %>%
  summarize(mean_abs_err = mean(abs_error)) %>%
  ggplot() +
  geom_point(aes(x = nodes, y = mean_abs_err, color = method)) +
  facet_wrap(~component, scales = "free")
  

  