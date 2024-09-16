library(ARTool)

# Load input data
df <- read.csv("C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\two_way_ART_ANOVA_input.csv")

# Convert variables to factors
df$Group <- as.factor(df$Group)
df$RM_Vals <- as.factor(df$RM_Vals)

# Construct the model
model <- art(Data ~ Group + RM_Vals + Group:RM_Vals, data = df)


# Perform the anova
# Perform the ANOVA
result <- anova(model)
# Store as a dataframe
result_output <- as.data.frame(result, row.name = make.unique(rownames(result)))


# Save the outputs
write.csv(result_output, "C:\\Users\\Jake\\Desktop\\python_code\\Lab_Analyses\\Lab_Analyses\\Utilities\\two_way_ART_ANOVA_anova.csv", row.names = FALSE)
