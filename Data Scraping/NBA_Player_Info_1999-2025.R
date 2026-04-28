library(hoopR)
library(dplyr)
library(purrr)

# (Important note: Please run this file locally.)
# Because this file uses the HoopR 2.1.0 package and built-in API equations,
# the file content will fluctuate continuously if you run again and again.
# Our analysis is based on the final run results from April 14.
# Running the program locally and, then uploading it.
# This fundamentally avoids the possibility of our model results crashing.

# seasons you want: 1999-01 to most recent completed season label
season_start <- 1999
season_end   <- most_recent_nba_season() - 1

season_labels <- sapply(season_start:season_end, year_to_season)

# pull one season safely
get_one_season <- function(season_label) {
  message("Downloading season: ", season_label)
  
  out <- tryCatch({
    tmp <- nba_leaguedashplayerstats(
      league_id = "00",
      season = season_label
    )
    
    # extract real table
    df <- tmp$LeagueDashPlayerStats
    
    # add season column
    df <- df %>%
      mutate(season = season_label)
    
    return(df)
  }, error = function(e) {
    message("Failed for season: ", season_label)
    return(NULL)
  })
  
  return(out)
}

# download all seasons and combine
all_player_stats <- map_dfr(season_labels, get_one_season)
all_player_stats %>% select(season, everything()) %>% View()

write.csv(all_player_stats, "all_player_stats_1999-2025.csv", row.names = FALSE)