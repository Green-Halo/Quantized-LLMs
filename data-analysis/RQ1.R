library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)

data <- read.csv("run_table_1.csv")

summary(data)

# Update quantization_type factor levels
data$quantization_type <- trimws(data$quantization_type)

# Update the order and levels of the factor
data$quantization_type <- factor(data$quantization_type, levels = c("32-bit", "16-bit", "awq-4-bit", "gptq-4-bit"))

# Plot GPU energy usage boxplot by quantization type
ggplot(data, aes(x = quantization_type, y = GPU.Energy, fill = quantization_type)) +
  geom_boxplot() +
  labs(title = "GPU Energy Usage by Quantization Type",
       x = "Quantization Type",
       y = "GPU Energy (J)") +
  theme_minimal() +
  theme(legend.position = "none",
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 11), # Y-axis title font size
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11), # X-axis title font size
        axis.text.x = element_text(size = 11))

# Calculate mean and standard deviation of energy consumption for each quantization type
energy_stats <- data %>%
  group_by(quantization_type) %>%
  summarise(
    mean_energy = mean(GPU.Energy, na.rm = TRUE),
    std_energy = sd(GPU.Energy, na.rm = TRUE)
  )

energy_stats

# Plot GPU energy usage bar chart by task and quantization type
energy_summary <- data %>%
  group_by(task_name, quantization_type) %>%
  summarise(mean_energy = mean(GPU.Energy, na.rm = TRUE), .groups = 'drop')

ggplot(energy_summary, aes(x = quantization_type, y = mean_energy, fill = quantization_type)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  geom_text(aes(label = sprintf("%.2f", mean_energy)), vjust = -0.5, position = position_dodge(width = 0.9)) +
  facet_wrap(~ task_name) +
  labs(title = "Comparison of GPU Energy Usage by Task and Quantization Type",
       x = "Quantization Type",
       y = "Mean GPU Energy (J)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 11), # Y-axis title font size
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11), # X-axis title font size
        axis.text.x = element_text(size = 11, angle = 45, hjust = 1)) # Rotate x-axis labels

# Normality test
shapiro_results <- data %>%
  group_by(quantization_type) %>%
  summarise(shapiro_stat = shapiro_test(GPU.Energy)$statistic,
            shapiro_p_value = shapiro_test(GPU.Energy)$p.value)

shapiro_results

# Normality test after transforming GPU.Energy
results <- data %>%
  group_by(quantization_type) %>%
  mutate(
    log_GPU_Energy = log(GPU.Energy + 1),  # Log transformation to handle zero values
    sqrt_GPU_Energy = sqrt(GPU.Energy)     # Square root transformation
  ) %>%
  group_by(quantization_type) %>%
  summarise(
    log_normality_p_value = shapiro_test(log_GPU_Energy)$p.value,
    sqrt_normality_p_value = shapiro_test(sqrt_GPU_Energy)$p.value
  )

results

# Kruskal-Wallis test
kruskal_test_result <- kruskal.test(GPU.Energy ~ quantization_type, data = data)

kruskal_test_result

# Calculate Cliff's Delta to quantify differences between quantization types
delta_32_16 <- cliff.delta(data$GPU.Energy[data$quantization_type == "32-bit"],
                           data$GPU.Energy[data$quantization_type == "16-bit"])

delta_32_awq4 <- cliff.delta(data$GPU.Energy[data$quantization_type == "32-bit"],
                             data$GPU.Energy[data$quantization_type == "awq-4-bit"])

delta_32_gptq4 <- cliff.delta(data$GPU.Energy[data$quantization_type == "32-bit"],
                              data$GPU.Energy[data$quantization_type == "gptq-4-bit"])

delta_16_awq4 <- cliff.delta(data$GPU.Energy[data$quantization_type == "16-bit"],
                             data$GPU.Energy[data$quantization_type == "awq-4-bit"])

delta_16_gptq4 <- cliff.delta(data$GPU.Energy[data$quantization_type == "16-bit"],
                              data$GPU.Energy[data$quantization_type == "gptq-4-bit"])

delta_awq4_gptq4 <- cliff.delta(data$GPU.Energy[data$quantization_type == "awq-4-bit"],
                                data$GPU.Energy[data$quantization_type == "gptq-4-bit"])

# Print results
print(list("32-bit vs 16-bit" = delta_32_16,
           "32-bit vs awq-4-bit" = delta_32_awq4,
           "32-bit vs gptq-4-bit" = delta_32_gptq4,
           "16-bit vs awq-4-bit" = delta_16_awq4,
           "16-bit vs gptq-4-bit" = delta_16_gptq4,
           "awq-4-bit vs gptq-4-bit" = delta_awq4_gptq4))