#import library to do http requests:
import urllib2
import urllib
import MySQLdb
import logging # means of tracking events that happen when some software runs. 

 
#import easy to use xml parser called minidom:
from xml.dom.minidom import parseString
from urllib2 import URLError

# Import from another python file to call  ( printMenue and readingFunction) function  ....
from MainFunctions import printMenue
from MainFunctions import readingXMLFile
from MainFunctions import getNewsFeedsURL
from RssDBFunctions import connectDB
from RssDBFunctions import createTable
from RssDBFunctions import insertValuesInDB
from RssDBFunctions import fetchFromServer

#--------------------------------------------- FUNCTION TO PRINT MENU ----------------

printMenue()

chossenFeed=0

while (chossenFeed != "11"):


        chossenFeed = raw_input("  Enter your favourite News feed: ");
    
        # ----- To get the NewsFeed category and the URL for xml file containing all news feeds
        urlLink, BBC_News_Feed_Name = getNewsFeedsURL(chossenFeed)  # return multiple values from getNewsFeedsURL Function 
        # --------------------------------------------------------------------------------------
      
      
        try:
             
            file = urllib2.urlopen(urlLink)
            
            
            # ------------------ Creating Database Table -----------------------------------------
            
            # should be called one time to not drop and re create table every time
            
            
            #createTable()  
           
            # ------------------------------------------------------------------------------------
            
            xmlDataTitle, xmlDataDescription = readingXMLFile (file,urlLink,BBC_News_Feed_Name)
            
           
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

fetchFromServer()






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