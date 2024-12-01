library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)

# Set the width and height of the plotting area in inches
options(repr.plot.width = 9, repr.plot.height = 9)

data <- read.csv("run_table_1.csv")

# Update factor levels for quantization_type
data$quantization_type <- trimws(data$quantization_type)

# Update the order of factor levels
data$quantization_type <- factor(data$quantization_type, levels = c("32-bit", "16-bit", "awq-4-bit", "gptq-4-bit"))

# Plot bar chart of CPU busy time by quantization type and task name
ggplot(data, aes(x = quantization_type, y = CPU.Busy.Time, fill = quantization_type)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.5) +
  labs(title = "CPU Busy Time by Quantization Type and Task Name",
       x = "Quantization Type", y = "CPU Busy Time (sec)", fill = "Quantization Type") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14, angle = 45, hjust = 1))+ # Rotate x-axis labels
  facet_wrap(~task_name, scales = "free")

# Plot bar chart of GPU busy time by quantization type and task name
ggplot(data, aes(x = quantization_type, y = GPU.Busy.Time, fill = quantization_type)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.5) +
  labs(title = "GPU Busy Time by Quantization Type and Task Name",
       x = "Quantization Type", y = "GPU Busy Time (sec)", fill = "Quantization Type") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14, angle = 45, hjust = 1))+ # Rotate x-axis labels
  facet_wrap(~task_name, scales = "free")

# Plot boxplot of memory usage by quantization type
ggplot(data, aes(x = quantization_type, y = Memory.Usage, fill = quantization_type)) +
  geom_boxplot() +
  geom_jitter(width = 0.2,height = 0.5, alpha = 0.5)+
  labs(title = "Memory Usage by Quantization Type",
       x = "Quantization Type",
       y = "Memory Usage (MB)") +
  theme_minimal() +
  theme(
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 14), # Y-axis title font size
        axis.text.y = element_text(size = 14),
        axis.title.x = element_text(size = 14), # X-axis title font size
        axis.text.x = element_text(size = 14))

# Normality test
shapiro_test_cpu <- data %>%
  group_by(quantization_type) %>%
  filter(length(unique(CPU.Busy.Time)) > 1) %>% # Filter out groups with all identical values
  shapiro_test(CPU.Busy.Time)
shapiro_test_cpu

shapiro_test_gpu <- data %>%
  group_by(quantization_type) %>%
  filter(length(unique(GPU.Busy.Time)) > 1) %>% # Filter out groups with all identical values
  shapiro_test(GPU.Busy.Time)
shapiro_test_gpu

shapiro_test_memory <- data %>%
  group_by(quantization_type) %>%
  filter(n() >= 1 & length(unique(Memory.Usage)) > 1) %>% # Filter out groups with sample size less than 3 or only one unique value of Memory.Usage
  # Add a check to ensure the group has data before applying shapiro_test
  do(if (nrow(.) > 0) shapiro_test(., Memory.Usage) else tibble(statistic = NA, p.value = NA)) %>%
  ungroup() # Ungroup to avoid potential issues later

shapiro_test_memory

# Kruskal-Wallis test
kruskal_test_cpu <- kruskal.test(CPU.Busy.Time ~ quantization_type, data = data)
kruskal_test_gpu <- kruskal.test(GPU.Busy.Time ~ quantization_type, data = data)
kruskal_test_memory <- kruskal.test(Memory.Usage ~ quantization_type, data = data)

kruskal_test_cpu
kruskal_test_gpu
kruskal_test_memory

# Calculate Cliff's Delta to quantify differences between quantization types
cliffs_delta_cpu <- cliff.delta(CPU.Busy.Time ~ quantization_type, data = data)
cliffs_delta_gpu <- cliff.delta(GPU.Busy.Time ~ quantization_type, data = data)
cliffs_delta_memory <- cliff.delta(Memory.Usage ~ quantization_type, data = data)

cliffs_delta_cpu
cliffs_delta_gpu
cliffs_delta_memory