library(shiny)
library(ggplot2)
library(dplyr)
library(ggfortify)
library(leaflet)
library(htmltools)
library(streamR)
library(RCurl)
library(RJSONIO)
library(stringr)
library(ROAuth)
library(twitteR)
library(base64enc)
library(tidyr)
library(rsconnect)
library(ggmap)

MyData1 <- read.csv(file = "data/dfloc.csv", stringsAsFactors = FALSE)
r_colors <- rgb(t(col2rgb(colors()) / 255))
names(r_colors) <- colors()


ui <- fluidPage(
  titlePanel(title=div(img(src = "lo.png", height = 100, width = 90), "Digital Crop Stress Surveillance")),
  sidebarLayout(
    sidebarPanel(radioButtons("gcountry", "Crop Stress Activity in:",
                              choices = c("Canada", "United States", "United Kingdom")),
                 selectInput("keyword", "Stress",
                             choices = c("Fusarium", "Septoria", "Yellow Rust", "Downy Mildew", "Powdery Mildew",
                                         "Snow Mold","Leaf Rust","Leaf Spot","Root Rot","Leaf Blight","Head Blight",
                                         "Early Blight","Late Blight","Leaf Curl","Brown Rust")),
                 selectInput("Category","category",
                             choices = c("Ad","Awareness","news","others","Research","self"),selected = "self"),
                  textInput("searchkw", label = "search:", value = "#Septoria")),
    mainPanel(
      tabsetPanel(
        tabPanel("Home",uiOutput("mySite"),htmlOutput("mysite1")),
        tabPanel("AgriTweets", leafletOutput("agmap",width = "80%",height = "400px")),
        tabPanel("Tweet Types", leafletOutput("map",width = "80%",height = "400px"),plotOutput("coolplot")),
        tabPanel("Live Tweets", tableOutput("op")),tabPanel("Live Map", leafletOutput("livemap"))),
        # plotOutput("coolplot"),
      br(), br(),
      tableOutput("results")
      
    )))


server <- function(input, output) {
  
  values <- reactiveValues(
    data = NULL
  )
  output$agmap <- renderLeaflet({
    icons <- icons(
      iconUrl = ifelse(MyData1$AgriTweet == 1,
                       'http://leafletjs.com/examples/custom-icons/leaf-green.png',
                       'http://leafletjs.com/examples/custom-icons/leaf-shadow.png'
      ),
      iconWidth = 38, iconHeight = 95,
      iconAnchorX = 22, iconAnchorY = 94,
      shadowUrl = "http://leafletjs.com/examples/custom-icons/leaf-shadow.png",
      shadowWidth = 50, shadowHeight = 64,
      shadowAnchorX = 4, shadowAnchorY = 62
    )
    print(icons)
    leaflet() %>%
      addTiles() %>%
      addMarkers(data = MyData1, lng = ~glang,lat = ~glat, icon = icons, popup = ~htmlEscape(tweet))
  })
  output$mySite <- renderUI({
  
    tags$a(href = "readme.txt", tags$img(src = "home.png", height = '80%', width = '80%'))
    # tags$img(src = "home.png")
  })
  
  consumer_key <- readLines("data/tokens.txt")[1]
  consumer_secret <- readLines("data/tokens.txt")[2]
  access_token <- readLines("data/tokens.txt")[3]
  access_secret <- readLines("data/tokens.txt")[4]
  options(httr_oauth_cache = TRUE) # enable using a local file to cache OAuth access credentials between R sessions
  setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)
  
  # Issue search query to Twitter
  dataInput <- reactive({  
    
    tweets <- twListToDF(searchTwitter(input$searchkw)) 
    View(tweets)
    temp_df <- twListToDF(lookupUsers(tweets$screenName))
    View(temp_df)
    
    temp_df1 <- merge(x = tweets, y = temp_df, by = "screenName", all = TRUE)
    View(temp_df1)
    temp_df2 <- as.character(temp_df1$location)
    # temp_df1[, 1][is.na(temp_df1[, 1])] <- "ua"
    tweets1 <- (geocode(temp_df2))
    tweets1[is.na(tweets1)]<-0
    View(tweets1)
    tweets$created <- as.character(tweets$created)
    tweets$longitude <- as.numeric(tweets1$lon)
    tweets$latitude <- as.numeric(tweets1$lat) 
    tweets$text <- as.character(tweets$text)
    tweets <- tweets[, c("text", "screenName", "created","longitude","latitude")]
  })
  
  mapTweets <- reactive({
    map = leaflet() %>% addTiles() %>%
      addMarkers(as.numeric(dataInput()$longitude), as.numeric(dataInput()$latitude), popup = dataInput()$text)
      
  })
  output$livemap = renderLeaflet(mapTweets())
  
  output$op <- renderTable(
    # print(dataInput())
    dataInput()[]
  )
  

  obsB <- observe({
    part1 <- input$gcountry
    part2 <- input$keyword
    part3 <- input$Category
    print(part1)
    print(part2)
    print(part3)
    
    MyData_new <- MyData1[MyData1$gcountry == part1,]
    MyData_new1 <- MyData_new[MyData_new$keyword == part2,]
    MyData_new2 <- MyData_new1[MyData_new1$Category == part3,]
    values$data <- MyData_new2
    
  })
  
  output$map <- renderLeaflet({
    # greenLeafIcon <- makeIcon(
    #   iconUrl = "http://leafletjs.com/examples/custom-icons/leaf-green.png",
    #   iconWidth = 38, iconHeight = 95,
    #   iconAnchorX = 22, iconAnchorY = 94,
    #   shadowUrl = "http://leafletjs.com/examples/custom-icons/leaf-shadow.png",
    #   shadowWidth = 50, shadowHeight = 64,
    #   shadowAnchorX = 4, shadowAnchorY = 62
    # )
    
    leaflet(data=MyData1) %>%
      addTiles() %>%
      addMarkers(data = values$data, lng = ~glang,lat = ~glat, popup = ~htmlEscape(tweet))
  })
  
  output$coolplot <- renderPlot({
    filtered <-
      MyData1 %>%
      filter(keyword == input$keyword,
             gcountry == input$gcountry)
    ggplot(filtered, aes(Category)) +
      geom_bar(aes(fill = Category))
  }, height = 400, width = 600)
  
  
  
}

getLocation <- function(x) {
  y <- getUser(x)
  location <- y$location
  return(location);
}

shinyApp(ui = ui, server = server)