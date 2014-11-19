#import library to do http requests:
import urllib2
import urllib
import MySQLdb
 
#import easy to use xml parser called minidom:
from xml.dom.minidom import parseString
from urllib2 import URLError

# Import from another python file to call  ( printMenue and readingFunction) function  ....
from YouTubeRead import printMenue
from YouTubeRead import readingXMLFile


#--------------------------------  FUNCTIONS NOT USED CAN BE MODIFIED TO BE USED FOR YOUTUPE API ---------

#from MainFunctions import getNewsFeedsURL
from RssDBFunctions import connectDB
from RssDBFunctions import createTable
from RssDBFunctions import insertValuesInDB
from RssDBFunctions import fetchFromServer

#--------------------------------------------- FUNCTION TO PRINT MENU ----------------

printMenue()

chossenFeed=0


  
       # urlLink, BBC_News_Feed_Name = getNewsFeedsURL(chossenFeed)  # return multiple values from getNewsFeedsURL Function 
        # --------------------------------------------------------------------------------------
urlLink= "http://gdata.youtube.com/feeds/api/standardfeeds/most_viewed?alt=rss"
#urlLink= "http://gdata.youtube.com/feeds/api/standardfeeds/most_recent?alt=rss"
#urlLink= "http://gdata.youtube.com/feeds/api/standardfeeds/most_discussed?alt=rss"
#urlLink = "http://gdata.youtube.com/feeds/api/standardfeeds/top_favorites?alt=rss"


# ========================= GETTING FEEDS FORM THE MOST POBULAR YOUTUBE VIDEOS IN SPECIFIC REGION "SA" 
urlLink = "https://gdata.youtube.com/feeds/api/standardfeeds/sa/most_popular?alt=rss&time=all_time"




      # https://gdata.youtube.com/feeds/api/videos/-/Foreign Language
      #URL notation:
      #https://gdata.youtube.com/feeds/api/videos/-/bass/fishing?v=2
      #category parameter:
      #https://gdata.youtube.com/feeds/api/videos?category=bass%2Cfishing&v=2
      
      #https://developers.google.com/youtube/2.0/developers_guide_protocol_video_feeds
      
    #URL for feed of most popular entertainment videos in France:
    #https://gdata.youtube.com/feeds/api/standardfeeds/FR/most_popular_Entertainment
      
      
try:
             
    file = urllib2.urlopen(urlLink)
            
            
            # ------------------ Creating Database Table -----------------------------------------
            
            # should be called one time to not drop and re create table every time
            
            
        #createTable()  
           
            # ------------------------------------------------------------------------------------
            
    xmlDataTitle = readingXMLFile (file,urlLink)
            
           
           # ------  insert values from xml file to RssFeeds Table
           #------------- moved to AllFunctions.py to insert all feeds there insted of inserting only the last feed
           #insertValuesInDB(BBC_News_Feed_Name, xmlDataTitle,xmlDataDescription,urlLink)
            
            
            
except urllib2.HTTPError, e:
            print "EXEPTION: ", e.code
except urllib2.URLError, e:
            print "EXEPTION:", e.code
except ValueError, e:
            print "EXEPTION:", e.args
except NameError, e:
            print "EXEPTION:", e.args


            # --------------READ Operation ----------
            

"""CREATE TABLE RSSFEEDS (
             FEED_CATEGORY  CHAR(100) NOT NULL,
             TAG_TITLE  CHAR(200) NOT NULL,
             DESCRIPTION  CHAR(200),
             URL_LINK VARCHAR(250)
"""








"""
url = 'http://10.0.2.2/android_connect/create_calls.php'
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
values = {'name' : 'Michael Foord',
          'location' : 'Northampton',
          'language' : 'Python' }
headers = { 'User-Agent' : user_agent }




data = urllib.urlencode(values)
req = urllib2.Request(url, data, headers)
response = urllib2.urlopen(req)
the_page = response.read()


"""


"""
from xml.dom import minidom


xmldoc = minidom.parse('rss.xml')
itemlist = xmldoc.getElementsByTagName('item') 
print len(itemlist)
print itemlist[0].attributes['title'].value
for s in itemlist :
    print s.attributes['title'].value
"""