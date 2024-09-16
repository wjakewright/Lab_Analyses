library(ARTool)
library(dplyr)

# Load the input data
df <- read.csv("C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\one_way_ART_input.csv")

df$Group <- as.factor(df$Group)

# Construct the model
model <- art(Data ~ Group + (1 | Rand_Var), data = df)

# Perform the ANOVA
result <- anova(model)
# Store as a dataframe
result_output <- as.data.frame(result, row.name = make.unique(rownames(result)))

# Save the outputs
write.csv(result_output, "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\one_way_ART_anova.csv", row.names = FALSE)
