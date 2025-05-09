
library(tidyverse)
library(cowplot)
library(latex2exp)
library(ggh4x)
save_path = '/Users/shiwen/Downloads/ggplots'
save_plot_as_pdf <- function(plot, folder, orientation = "landscape") {
  # Extract the title from the plot
  title <- ggplot_build(plot)$plot$labels$title
  
  # Handle cases where no title is set
  if (is.null(title) || title == "") {
    title <- "untitled_plot"
  }
  
  # Clean title to make it a valid filename
  title <- gsub("[^a-zA-Z0-9_]", "_", title)  # Replace invalid characters with "_"
  
  # Define the full file path
  file_path <- file.path(folder, paste0(title, ".pdf"))
  
  # Set A4 dimensions based on orientation
  if (orientation == "landscape") {
    width <- 11.69
    height <- 8.27
  } else {  # Default to portrait
    width <- 8.27
    height <- 11.69
  }
  
  # Save the plot as a PDF
  ggsave(file_path, plot = plot, device = "pdf", width = width, height = height, units = "in")
  
  message("Plot saved as: ", file_path)
}


# evolution of the model example ------------------------------------------
evo <- read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/time_vs_lat_pos/neg_4_sample.csv") %>% as_tibble() %>% select(-X)


# Comparing Alignment: Oracle vs. RGD -------------------------------------
Oracle_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/oracle_performance.csv")
RGD_performance = read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/est_lat_pos/example_1/convergence/RGD_performance.csv")

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




# all_lat_pos %>% 
#   filter(time == 1) %>%
#   ggplot()+
#   geom_point(aes(x = dim_1, y = dim_2, color = group), size = 0.1) +
#   facet_wrap(~method, scale = "free")
  


# Estimator Consistency and Variance --------------------------------------

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
  select(-value) %>%
  filter(number_of_iterations != max_iterations)

comparison_summary <- consistency_both %>% 
  group_by(nodes, seed, init, method) %>%
  reframe(C_B_est = c(H %*% B_est),
          C_B_real = c(H %*% B_real),
          component =  paste0("beta", 1:4),
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



# Empirical Variance vs. Theoretical Variance -----------------------------

path_B_est <- "/Users/shiwen/Desktop/ABC data sets/consistency/est_all.csv"
B_est <- read.csv(path_B_est) %>% na.omit() %>% as_tibble() %>% filter(number_of_iterations != max_iterations)
constraint <- c(1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -0., -0.,  0.,  0.,  0.,
                0.,  0.,  1.,  0.,  0.,  0., -1., -0., -0.,  0.,  0.,  1.,  0.,  0.,
                0.,  0.,  0.,  0., -0., -1., -0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
                0.,  0., -0., -1., -0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                -0., -0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -0., -0.,
                -1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.) %>% matrix(4, 21)

  
B_est_bias <- B_est %>% 
  select(-X) %>%
  pivot_longer(cols = starts_with("method_"),
               values_to = "value",
               names_to = "method",
               names_prefix = "method_") %>%
  filter(value == 1) %>%
  group_by(seed, nodes, method) %>%
  reframe(C_B_est = c(solve(tcrossprod(constraint), constraint %*% B_est)),
          C_B_real = c(solve(tcrossprod(constraint), constraint %*% B_real)),
          component = paste0("beta", 1:4),
          time_elapsed = mean(time_elapsed)) %>%
  mutate(Error = (C_B_est - C_B_real)) %>%
  group_by(nodes, method, component) %>%
  summarize(Bias = mean(Error),
            Mean_ABS_Error = mean(abs(Error)),
            SE = sd(Error),
            Mean_Time_Elapsed = mean(time_elapsed),
            SD_Time_Elapsed = sd(time_elapsed)) 




path_fish_est <- "/Users/shiwen/Desktop/ABC data sets/consistency/fish_all.csv"
fish_est <- read.csv(path_fish_est) %>% na.omit() %>% as_tibble()
fish_est_long <- fish_est %>%
  rename(id = "X") %>%
  pivot_longer(cols = starts_with("method_"),
               values_to = "value",
               names_to = "method",
               names_prefix = "method_") %>%
  filter(value == 1) %>%
  select(-value) %>%
  mutate(method = case_when(
    method == 1 ~ "OL",
    method == 2 ~ "OA", 
    method == 3 ~ "NO"
  )) %>%
  pivot_longer(cols = starts_with("fisher_info_"),
               values_to = "fisher_info",
               names_to = "component",
               names_prefix = "fisher_info_")


fish_summary <- fish_est_long %>% 
  group_by(nodes, method, id) %>%
  reframe(diag_inv_H_fish_tH = c(diag(H %*% solve(matrix(fisher_info, 21, 21) )%*% t(H))),
          component = paste0("beta",1:4)) %>%
  mutate(st_dev = sqrt(diag_inv_H_fish_tH)) %>%
  group_by(nodes, method, component) %>%
  summarize(st_dev = mean(st_dev))


B_SE_STD <- full_join(fish_summary, B_est_bias, by = join_by(nodes == nodes, method == method, component == component))


# Robustness for Embedding Dimension Misspecification ---------------------
path_robustness <- "/Users/shiwen/Documents/GitHub/ABCDPRGM/simulated_data/dimension_robustness/robustness.csv"
df_robust <- read.csv(path_robustness) %>% 
  as_tibble() %>%
  mutate(beta1 = beta1 - 1, 
         beta2 = beta2 - 1, 
         beta3 = beta3 + 4, 
         beta4 = beta4 - 5) %>% 
  na.omit %>%
  pivot_longer(cols = starts_with("beta"), 
               names_to = "component",
               names_prefix = "beta") %>%
  group_by(n, p0, component) %>%
  summarize(m = mean(value), m_info = mean(info_lost),sd = sd(value))
# real data: away and toward group in action -----------------------------------------
aw_bias <- read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/real_data/aw_bias.csv")[,-1] %>% as_tibble()
tw_bias <- read.csv("/Users/shiwen/Documents/GitHub/ABCDPRGM/real_data/tw_bias.csv")[,-1] %>% as_tibble()



# real data: away group eigenvalues ---------------------------------------
aw_eigval <- read.csv("/Users/shiwen/Documents/GitHub/Python Projects/ABCDPRGM/real_data/AW_eigval.csv") %>% 
  as_tibble() %>%
  select(-X, -index)


# Plot theme --------------------------------------------------------------
theme_big <- function() {
  theme(
    text = element_text(size = 16, color = "black"),          # Default text size, slightly smaller for paper
    axis.title.x = element_text(size = 20, face = "bold", margin = margin(t = 10, b = 10, unit = "pt")),  # Compact margins for paper
    axis.title.y = element_text(size = 20, face = "bold", margin = margin(l = 10, r = 10, unit = "pt")),  # Compact y-axis labels
    axis.text = element_text(size = 16),                      # Tick labels small and readable
    legend.title = element_text(size = 14, face = "bold"),    # Legend title size appropriate for paper
    legend.text = element_text(size = 14),                    # Smaller legend text
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5, margin = margin(t = 20, b = 20, unit = "pt")),  # Title slightly larger but not too much
    plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt")  # Minimal margins to save space
  )
}

# Plot 0: evolution of the model -- Example 1 -----------------------------

plot_0 <- evo %>% filter(time %in% c(6, 8, 10, 12)) %>% 
  mutate(Group = as.factor(group)) %>%
  ggplot(aes(x = dim_1, y = dim_2, color = Group)) +
  geom_point(size = 1, alpha = 0.3) +
  geom_density_2d(linewidth = 0.5, alpha = 0.8)  + 
  facet_wrap(~time, labeller = labeller(time = c('6' = 'T = 6', '8' = 'T = 8', '10' = 'T = 10', '12' = 'T = 12')))+
  labs(x = "Dimension 1", y = "Dimension 2", title = "Polarization Through Time") +
  theme_big()

save_plot_as_pdf(plot_0, save_path)


# Plot 1: Align Error: Oracle vs. RGD -- Example 1 ----------------------------------------------------------------

df_performance %>%
  ggplot(aes(x = nodes, y = Error, color = Time, shape = method)) +
  geom_point(size = 6) +
  labs(
    title = "Align Error: Oracle vs. RGD -- Example 1",
    x = "Nodes",
    y = "Error",
    color = "Time",
    shape = "Method Used to Align"
  ) +
  theme_big() +
  scale_shape_manual(values = c(3,4))
  


# Plot 2: Align Overview: Example 1 ---------------------------------------

all_lat_pos %>% 
  filter(time == 0) %>%
  ggplot()+
  geom_point(aes(x = dim_1, y = dim_2, color = group), size = 0.1) +
  facet_wrap(~method, scale = "free")+
  labs(
    title = "Align Overview: Example 1",
    x = "Dimension 1",
    y = "Dimension 2",
    color = "Group"
  ) +
  theme_big()



# Plot 3: Comparing Estimates from Different Initialization ---------------


my_labeller <- as_labeller(c("beta1" ="\U03B2[1]", "beta2" ="\U03B2[2]", "beta3" ="\U03B2[3]","beta4" ="\U03B2[4]"),
                           default = label_parsed)
comparison_summary %>% 
  ggplot(aes(x = nodes, y = bias, ymin = bias - 2 * sd_error, ymax = bias + 2 * sd_error, color = method, shape = init)) +
  geom_pointrange(size = 1, position = position_dodge(width =  450)) +
  facet_wrap(~component, scales = "free", 
             labeller = my_labeller) +
  labs(
    title = paste0("Comparing Estimates from Different Initialization: ", "Estimation Error", " \U00B1 ", "2 SE" ),
    x = "Nodes",
    y = "Estimation Error",
    color = "Method",
    shape = "Initialization"
  ) +
  theme_big()

# Plot 4: Comparing Run Time When Using Different Initialization ----------


comparison_summary %>%
  ggplot(aes(x = nodes, y = mean_time_elapsed, ymin = mean_time_elapsed - 2 * sd_time_elapsed, ymax = mean_time_elapsed + 2 * sd_time_elapsed,color = init, shape = method)) + 
  geom_pointrange(position = position_dodge(width = 200), size = 1)+
  labs(
    title = "Comparing Run Time When Using Different Initialization",
    x = "Nodes",
    y = "Time: Seconds",
    color = "Initialization",
    shape = "Method"
  ) +
  theme_big()



# Plot 5: Comparing Bias of Different Methods -----------------------------

B_est_bias %>%
  ggplot(aes(x = nodes, y = Bias, color = method)) +
  geom_pointrange(aes(ymin = Bias - 2*SE, ymax = Bias + 2*SE), position = position_dodge(width = 450)) +
  facet_wrap(~component, scales = "free", labeller = my_labeller)+
  labs(
    title = paste0("Comparing Estimation Error of Different Methods: ", "Estimation Error", " \U00B1 ", "2 SE" ),
    x = "Nodes",
    y = "Estimation Error",
    color = "Method",
    shape = "Initialization"
  ) +
  theme_big()

  


# Plot 6: Nodes vs. ratio of Empirical and Theoretical Standard Deviation -----------------------------------------------------------------

my_labeller <- as_labeller(c("beta1" ="\U03B2[1]", "beta2" ="\U03B2[2]", "beta3" ="\U03B2[3]","beta4" ="\U03B2[4]"),
                           default = label_parsed)
B_SE_STD %>%
  mutate(ratio = SE/st_dev) %>%
  ggplot(aes(x = nodes, y = ratio, color = method)) +
  geom_point(size = 3) + 
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 1)+
  facet_wrap(~component, scales = "free", labeller = my_labeller) +
  labs(
    title = paste0("Nodes vs. ratio of Empirical and Theoretical Standard Deviation"),
    x = "Nodes",
    y = "Ratio",
    color = "Method"
  ) +
  theme_big()


# Plot 7: Nodes vs. Theoretical/Empirical Standard Deviation----------------------------------------------------------------

my_labeller <- as_labeller(c("beta1" ="\U03B2[1]", "beta2" ="\U03B2[2]", "beta3" ="\U03B2[3]","beta4" ="\U03B2[4]", 
                             "OL" = "OL", "OA" = "OA", "NO" = "NO"),
                           default = label_parsed)
B_SE_STD %>%
  select(nodes, method, component, st_dev, SE) %>%
  rename(Theoretical = "st_dev",
         Empirical = "SE") %>%
  pivot_longer(cols = c("Theoretical", "Empirical"),
               values_to = "STD_EST",
               names_to = "STD_EST_type") %>%
  ggplot() +
  geom_point(aes(x = nodes, y = STD_EST, color = STD_EST_type), size = 2) + 
  ggh4x::facet_grid2(method ~ component, scales = "free_y", independent = "y", labeller = my_labeller) + 
  labs(
    title = "Nodes vs. Theoretical/Empirical Standard Deviation, by Component and Method",
    x = "Nodes",
    y = "Standard Deviation",
    color = "",
    shape = "Method"
  ) +
  theme_big()
# Plot 8: Run Time Comparison Across Methods --------------------------------------

B_est_bias %>%
  ggplot(aes(x = nodes, y = Mean_Time_Elapsed, color = method)) +
  geom_pointrange(aes(ymin = Mean_Time_Elapsed - 2*SD_Time_Elapsed, ymax = Mean_Time_Elapsed + 2*SD_Time_Elapsed), position = position_dodge(width = 450))+
  labs(
    title = "Comparing Run Time When Using Different Methods",
    x = "Nodes",
    y = "Time: Seconds",
    color = "Method",
    shape = "Initialization"
  ) +
  theme_big()







B_SE_STD %>%
  select(nodes, method, component, st_dev, SE) %>%
  rename(Theoretical = "st_dev",
         Empirical = "SE") %>%
  pivot_longer(cols = c("Theoretical", "Empirical"),
               values_to = "STD_EST",
               names_to = "STD_EST_type") %>%
  filter(method != "OA") %>% 
  ggplot(aes(x = nodes, y = STD_EST, shape = STD_EST_type, color = method, linetype = STD_EST_type)) +
  # geom_point(size = 2, alpha = .5) + #, position= position_dodge(width = 400)) + 
  geom_line() +
  facet_wrap(~component, scales = "free")+
  theme_big()





# Plot 9: Embedding Dimension vs. Mean Bias for Robustness -----------------

df_robust %>% 
  ggplot(aes(x = p0, y = m, color = factor(n))) +
  geom_pointrange(aes(ymin = m - 2 * sd, ymax = m + 2 * sd), 
                  position = position_dodge(width = 0.2)) +
  facet_wrap("component", scales = "free") +
  labs(
    title = "Bias with ±2 SD Error Ranges for the 4 Components of β",
    x = "Embedding Dimension",
    y = "Mean Bias (m)",
    color = "n"
  ) +
  theme_big()
# Plot 10: Embedding Dimension vs. Info Lost for Robustness ---------------
df_robust %>% 
  ggplot(aes(x = p0, y = m_info, color = factor(n))) +
  geom_line() + 
  labs(
    title = "Embedding Dimension vs. Info Lost (True Dimension = 5)",
    x = "Embedding Dimension",
    y = "Percentage Info Lost",
    color = "Number of Nodes"
  ) +
  theme_big()

# Plot 11: Away: est vs. dim with error bars (Real Data) ---------------------------------------------
aw_bias %>% pivot_longer(
    cols = starts_with("b"),
    names_to = "est_comp",
    names_prefix = "b",
    values_to = "est"
  ) %>%
  pivot_longer(
    cols = starts_with("s"),
    names_to = "SD_comp",
    names_prefix = "s",
    values_to = "SD"
  ) %>% 
  filter(est_comp == SD_comp) %>%
  select(-SD_comp) %>%
  mutate(est_comp = as.numeric(est_comp) + 1) %>%
  ggplot(aes(x = dim, y = est)) +
  geom_point() +  # Scatter plot
  geom_errorbar(aes(ymin = est - 2 * SD, ymax = est + 2 * SD), width = 0.1) +  # Error bars with 2*SD
  facet_wrap(~ est_comp, scales = "free") +  # Facet wrap by `est_comp`
  labs(x = "Dimension", y = "Estimate", title = "Away: Dim vs. Est with Error Bar (2*SD)") +
  theme_big()


# Plot 12: Toward: est vs. dim with error bars (Real Data) ---------------------------------------------
tw_bias %>% pivot_longer(
  cols = starts_with("b"),
  names_to = "est_comp",
  names_prefix = "b",
  values_to = "est"
) %>%
  pivot_longer(
    cols = starts_with("s"),
    names_to = "SD_comp",
    names_prefix = "s",
    values_to = "SD"
  ) %>% 
  filter(est_comp == SD_comp) %>%
  select(-SD_comp) %>%
  mutate(est_comp = as.numeric(est_comp) + 1) %>%
  ggplot(aes(x = dim, y = est)) +
  geom_point() +  # Scatter plot
  geom_errorbar(aes(ymin = est - 2 * SD, ymax = est + 2 * SD), width = 0.1) +  # Error bars with 2*SD
  facet_wrap(~ est_comp, scales = "free") +  # Facet wrap by `est_comp`
  labs(x = "Dimension", y = "Estimate", title = "Toward: Dim vs. Est with Error Bar (2*SD)") +
  theme_big()



# Plot 13: Away Eigenvalues -----------------------------------------------
eigval <- aw_eigval %>% 
  pivot_longer(cols = starts_with("Eigen"),
               names_prefix = "Eigenvalue_",
               names_to = "time",
               values_to = "eigenvalue") %>%
  mutate(Signs = if_else(eigenvalue > 0, "positive", "negative"),
         time = if_else(time == 0, "t = 0", "t = 1")) %>%
  group_by(time) %>%
  arrange(desc(eigenvalue)) %>%
  mutate(rank = row_number())

eigval %>%
  ggplot(aes(x = rank, y = abs(eigenvalue), color = Signs))+
  geom_point(size = 2) +
  facet_wrap(~time, scales = "free_x")+
  labs(x = "Rank (Descending Order)",
       y = "Eigenvalue",
       title = "Eigenvalues in Descending Order by Time") +
  theme_big()

