library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)

# Set the width and height of the plotting area in inches
options(repr.plot.width = 9, repr.plot.height = 9)

data <- read.csv("run_tables/llama.csv")

summary(data)

# Update factor levels for quantization_type
data$quantization_type <- trimws(data$quantization_type)

# Update the order of factor levels
data$quantization_type <- factor(data$quantization_type, levels = c("32-bit", "16-bit", "awq-4-bit", "gptq-4-bit"))

# Plot boxplot of accuracy by quantization type
ggplot(data, aes(x = quantization_type, y = Accuracy, fill = quantization_type)) +
  geom_boxplot() +
  geom_jitter(width = 0.2, height = 0.002, alpha = 0.5) +
  labs(title = "Accuracy by Quantization Type",
       x = "Quantization Type",
       y = "Accuracy (%)") +
  theme_minimal() +
  theme(
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14))

# Calculate mean and standard deviation of energy consumption for each quantization type
energy_stats <- data %>%
  group_by(quantization_type) %>%
  summarise(
    mean_energy = mean(GPU.Energy, na.rm = TRUE),
    std_energy = sd(GPU.Energy, na.rm = TRUE)
  )

energy_stats

# Plot bar chart of accuracy by task and quantization type
mean_accuracy <- data %>%
  group_by(quantization_type, task_name) %>%
  summarise(mean_accuracy = mean(Accuracy, na.rm = TRUE), .groups = 'drop')

# Create bar chart
ggplot(mean_accuracy, aes(x = quantization_type, y = mean_accuracy, fill = task_name)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  geom_text(aes(label = sprintf("%.2f", mean_accuracy)), vjust = -0.5, position = position_dodge(width = 0.9)) +
  labs(title = "Comparison of Accuracy by Quantization Type and Task",
       x = "Quantization Type",
       y = "Mean Accuracy") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14, angle = 45, hjust = 1)) # Rotate x-axis labels

# Normality test
shapiro_results <- data %>%
  group_by(quantization_type) %>%
  summarise(shapiro_stat = shapiro_test(Accuracy)$statistic,
            shapiro_p_value = shapiro_test(Accuracy)$p.value)

shapiro_results

# Kruskal-Wallis test
kruskal_test_result <- kruskal.test(Accuracy ~ quantization_type, data = data)

kruskal_test_result

# Calculate Cliff's Delta to quantify differences between quantization types
delta_32_16 <- cliff.delta(data$Accuracy[data$quantization_type == "32-bit"],
                           data$Accuracy[data$quantization_type == "16-bit"])

delta_32_awq4 <- cliff.delta(data$Accuracy[data$quantization_type == "32-bit"],
                             data$Accuracy[data$quantization_type == "awq-4-bit"])

delta_32_gptq4 <- cliff.delta(data$Accuracy[data$quantization_type == "32-bit"],
                              data$Accuracy[data$quantization_type == "gptq-4-bit"])

delta_16_awq4 <- cliff.delta(data$Accuracy[data$quantization_type == "16-bit"],
                             data$Accuracy[data$quantization_type == "awq-4-bit"])

delta_16_gptq4 <- cliff.delta(data$Accuracy[data$quantization_type == "16-bit"],
                              data$Accuracy[data$quantization_type == "gptq-4-bit"])

delta_awq4_gptq4 <- cliff.delta(data$Accuracy[data$quantization_type == "awq-4-bit"],
                                data$Accuracy[data$quantization_type == "gptq-4-bit"])

# Output results
print(list("32-bit vs 16-bit" = delta_32_16,
           "32-bit vs awq-4-bit" = delta_32_awq4,
           "32-bit vs gptq-4-bit" = delta_32_gptq4,
           "16-bit vs awq-4-bit" = delta_16_awq4,
           "16-bit vs gptq-4-bit" = delta_16_gptq4,
           "awq-4-bit vs gptq-4-bit" = delta_awq4_gptq4))