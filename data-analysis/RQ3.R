library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)

data <- read.csv("run_table_1.csv")

# Update quantization_type factor levels
data$quantization_type <- trimws(data$quantization_type)

# Update the order and levels of the factor
data$quantization_type <- factor(data$quantization_type, levels = c("32-bit", "16-bit", "awq-4-bit", "gptq-4-bit"))

# Plot CPU and GPU busy times bar chart by quantization type and task name
ggplot(data, aes(x = quantization_type)) +
  geom_bar(aes(y = CPU.Busy.Time, fill = "CPU Busy Time"), stat = "identity", width = 0.3, position = position_nudge(x = -0.15)) +
  geom_bar(aes(y = GPU.Busy.Time, fill = "GPU Busy Time"), stat = "identity", width = 0.3, position = position_nudge(x = 0.15)) +
  labs(title = "Comparison of CPU and GPU Busy Times by Quantization Type and Task Name",
       x = "Quantization Type", y = "Busy Time (sec)", fill = "Busy Time Type: ") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 11),
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11),
        axis.text.x = element_text(size = 11, angle = 45, hjust = 1)) +
  facet_wrap(~task_name, scales = "free")

# Plot memory usage boxplot by quantization type
ggplot(data, aes(x = quantization_type, y = Memory.Usage, fill = quantization_type)) +
  geom_boxplot() +
  labs(title = "Memory Usage by Quantization Type",
       x = "Quantization Type",
       y = "Memory Usage (MB)") +
  theme_minimal() +
  theme(legend.position = "none",
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 11), # Y-axis title font size
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11), # X-axis title font size
        axis.text.x = element_text(size = 11, angle = 45, hjust = 1))

# Normality test for CPU busy time
shapiro_test_cpu <- data %>%
  group_by(quantization_type) %>%
  filter(length(unique(CPU.Busy.Time)) > 1) %>% # Filter out groups with all identical values
  shapiro_test(CPU.Busy.Time)
shapiro_test_cpu

# Normality test for GPU busy time
shapiro_test_gpu <- data %>%
  group_by(quantization_type) %>%
  filter(length(unique(GPU.Busy.Time)) > 1) %>% # Filter out groups with all identical values
  shapiro_test(GPU.Busy.Time)
shapiro_test_gpu

# Normality test for memory usage
shapiro_test_memory <- data %>%
  group_by(quantization_type) %>%
  filter(length(unique(Memory.Usage)) > 1) %>% # Filter out groups with all identical values
  shapiro_test(Memory.Usage)
shapiro_test_memory

# Kruskal-Wallis test for CPU busy time
kruskal_test_cpu <- kruskal.test(CPU.Busy.Time ~ quantization_type, data = data)
# Kruskal-Wallis test for GPU busy time
kruskal_test_gpu <- kruskal.test(GPU.Busy.Time ~ quantization_type, data = data)
# Kruskal-Wallis test for memory usage
kruskal_test_memory <- kruskal.test(Memory.Usage ~ quantization_type, data = data)

kruskal_test_cpu
kruskal_test_gpu
kruskal_test_memory

# Calculate Cliff's Delta to quantify differences between quantization types for CPU busy time
cliffs_delta_cpu <- cliff.delta(CPU.Busy.Time ~ quantization_type, data = data)
# Calculate Cliff's Delta to quantify differences between quantization types for GPU busy time
cliffs_delta_gpu <- cliff.delta(GPU.Busy.Time ~ quantization_type, data = data)
# Calculate Cliff's Delta to quantify differences between quantization types for memory usage
cliffs_delta_memory <- cliff.delta(Memory.Usage ~ quantization_type, data = data)

cliffs_delta_cpu
cliffs_delta_gpu
cliffs_delta_memory