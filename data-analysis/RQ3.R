library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)
library(gridExtra)
library(repr)

# Set the width and height of the plotting area in inches
options(repr.plot.width = 9, repr.plot.height = 9)

data <- read.csv("run_tables/llama.csv")

summary(data)

# Update factor levels for quantization_type
data$quantization_type <- trimws(data$quantization_type)

# Update the order of factor levels
data$quantization_type <- factor(data$quantization_type, levels = c("32-bit", "16-bit", "awq-4-bit", "gptq-4-bit"))

# Plot boxplot of inference time by quantization type
ggplot(data, aes(x = quantization_type, y = Inference.Time, fill = quantization_type)) +
  geom_boxplot() +
  geom_jitter(width = 0.2, height = 0.5, alpha = 0.5) +
  labs(title = "Inference Time by Quantization Type",
       x = "Quantization Type",
       y = "Inference Time (sec)") +
  theme_minimal() +
  theme(
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14))

# Modify plot_by_quantization function to adjust aspect.ratio
plot_by_quantization <- function(quant_type) {
  df <- subset(data, quantization_type == quant_type)

  # Calculate average for each task
  avg_data <- aggregate(`Inference.Time` ~ task_name, data = df, mean)

  ggplot(df, aes(x = task_name, y = `Inference.Time`, color = task_name)) +
    geom_point(size = 2, show.legend = FALSE, position = position_jitter(width = 0.2, height = 0)) +  # Jitter points to avoid overlap
    geom_point(data = avg_data, aes(x = task_name, y = `Inference.Time`), color = "black", shape = 4, size = 5) +  # Cross marker for average
    geom_text(data = avg_data, aes(x = task_name, y = `Inference.Time`, label = round(`Inference.Time`, 2)), vjust = 0, hjust = 1.2, color = "black", size = 4) +  # Show average values
    labs(title = quant_type, x = "Task Name", y = "Inference Time (sec)") +
    theme_minimal() +
    theme(legend.position = "bottom",
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14, angle = 45, hjust = 1)) # Rotate x-axis labels
}

options(repr.plot.width = 20, repr.plot.height = 5)
# Regenerate four plots
p1 <- plot_by_quantization('32-bit')
p2 <- plot_by_quantization('16-bit')
p3 <- plot_by_quantization('awq-4-bit')
p4 <- plot_by_quantization('gptq-4-bit')

# Arrange scatter plots in a row of four columns
grid.arrange(p1, p2, p3, p4, ncol = 4)

# Normality test
shapiro_test_results <- data %>%
  group_by(quantization_type) %>%
  filter(length(unique(Inference.Time)) > 1) %>% # Filter out groups with all identical values
  shapiro_test(Inference.Time)

shapiro_test_results

# Normality test after log and square root transformation
results <- data %>%
  group_by(quantization_type) %>%
  mutate(
    log_Inference_Time = log(Inference.Time + 1),  # Log transformation to handle zero values
    sqrt_Inference_Time = sqrt(Inference.Time)     # Square root transformation
  ) %>%
  group_by(quantization_type) %>%
  filter(length(unique(log_Inference_Time)) > 1 & length(unique(sqrt_Inference_Time)) > 1) %>% # Filter out groups with all identical values
  summarise(
    log_normality_p_value = shapiro_test(log_Inference_Time)$p.value,
    sqrt_normality_p_value = shapiro_test(sqrt_Inference_Time)$p.value
  )

results

# Kruskal-Wallis test
kruskal_test_result <- kruskal.test(Inference.Time ~ quantization_type, data = data)

kruskal_test_result

# Calculate Cliff's Delta to quantify differences between quantization types
delta_32_16 <- cliff.delta(data$Inference.Time[data$quantization_type == "32-bit"],
                           data$Inference.Time[data$quantization_type == "16-bit"])

delta_32_awq4 <- cliff.delta(data$Inference.Time[data$quantization_type == "32-bit"],
                             data$Inference.Time[data$quantization_type == "awq-4-bit"])

delta_32_gptq4 <- cliff.delta(data$Inference.Time[data$quantization_type == "32-bit"],
                              data$Inference.Time[data$quantization_type == "gptq-4-bit"])

delta_16_awq4 <- cliff.delta(data$Inference.Time[data$quantization_type == "16-bit"],
                             data$Inference.Time[data$quantization_type == "awq-4-bit"])

delta_16_gptq4 <- cliff.delta(data$Inference.Time[data$quantization_type == "16-bit"],
                              data$Inference.Time[data$quantization_type == "gptq-4-bit"])

delta_awq4_gptq4 <- cliff.delta(data$Inference.Time[data$quantization_type == "awq-4-bit"],
                                data$Inference.Time[data$quantization_type == "gptq-4-bit"])

# Output results
print(list("32-bit vs 16-bit" = delta_32_16,
           "32-bit vs awq-4-bit" = delta_32_awq4,
           "32-bit vs gptq-4-bit" = delta_32_gptq4,
           "16-bit vs awq-4-bit" = delta_16_awq4,
           "16-bit vs gptq-4-bit" = delta_16_gptq4,
           "awq-4-bit vs gptq-4-bit" = delta_awq4_gptq4))