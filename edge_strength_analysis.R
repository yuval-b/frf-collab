# Quick look at the distribution of edge strengths produced by
# networking_matcher_claude.py, bucketed the same way as the "Strength of
# Edges" slide handed out on the night.
#
# Requirements: install.packages(c("tidyverse"))
# Usage: Rscript edge_strength_analysis.R   (run from the repo root)

library(tidyverse)

edges <- read_csv("output/edges_ranked.csv") %>%
  mutate(
    Strength = case_when(
      total > 0.40                 ~ "Very strong (0.4+)",
      total > 0.32 & total <= 0.40 ~ "Strong (0.32-0.4)",
      total > 0.24 & total <= 0.32 ~ "Moderate (0.24-0.32)",
      total <= 0.24                ~ "Weak"
    ),
    Strength = factor(
      Strength,
      levels = c("Very strong (0.4+)", "Strong (0.32-0.4)", "Moderate (0.24-0.32)", "Weak")
    )
  )

p <- ggplot(edges) +
  geom_histogram(aes(x = total, fill = Strength), breaks = seq(0, 0.65, 0.01)) +
  theme_minimal() +
  scale_fill_manual(values = c(
    "Very strong (0.4+)"   = "#2b784f",
    "Strong (0.32-0.4)"    = "#7cae37",
    "Moderate (0.24-0.32)" = "#cbdb3f",
    "Weak"                 = "#849da9"
  )) +
  xlab("Edge Strength") +
  ylab("Count of edges")

ggsave("output/edge_strength_histogram.png", p, width = 8, height = 5, dpi = 150)
