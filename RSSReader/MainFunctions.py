#import library to do http requests:
import urllib2
import urllib


#--------------
#============
#checking the SQL #checking the SQL #checking the SQL #checking the SQL 
#import MySQLdb
 
#import easy to use xml parser called minidom:
from xml.dom.minidom import parseString
from RssDBFunctions import insertValuesInDB

#--------------------------------------------- FUNCTION TO PRINT MENU ----------------
def printMenue():
   # Add both the parameters and return them."

    print " \n "
    print " ---------------------------------  "
    print " |  1- Top Stories               | "
    print " |  2- World                     | "
    print " |  3- UK                        | "
    print " |  4- Business                  | "
    print " |  5- Politics                  | "
    print " |  6- Health                    | "
    print " |  7- Education & Family        | "
    print " |  8- Science & Environment     | "
    print " |  9- Technology                |"
    print " |  10- Entertainment & Arts     | "
    print " ---------------------------------  "
    return 


 # --------------- FUNCTION TO READ and from XML URL and PRINT ( TITLE - DESCRIPTION - LINK ) here



def getNewsFeedsURL( chossenFeed ):
    
    urlLink = ""
    BBC_News_Feed_Name =""
    
    if chossenFeed == "1":
            urlLink = "http://feeds.bbci.co.uk/news/rss.xml"
            BBC_News_Feed_Name ="TopStories"
    elif chossenFeed =="2":
             urlLink = "http://feeds.bbci.co.uk/news/world/rss.xml"
             BBC_News_Feed_Name ="World"
    elif chossenFeed=="3":
             urlLink = "http://feeds.bbci.co.uk/news/uk/rss.xml"
             BBC_News_Feed_Name ="UK"
    elif chossenFeed =="4":
             urlLink = "http://feeds.bbci.co.uk/news/business/rss.xml"
             BBC_News_Feed_Name ="Business"
    elif chossenFeed =="5":
            urlLink = "http://feeds.bbci.co.uk/news/politics/rss.xml"
            BBC_News_Feed_Name ="Politics"
    elif chossenFeed =="6":
            urlLink = "http://feeds.bbci.co.uk/news/health/rss.xml"
            BBC_News_Feed_Name ="Health"
    elif chossenFeed =="7":
            urlLink = "http://feeds.bbci.co.uk/news/education/rss.xml"
            BBC_News_Feed_Name ="EducationFamily"
    elif chossenFeed =="8":
            urlLink = "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"
            BBC_News_Feed_Name ="ScienceEnvironment"
    elif chossenFeed =="9":
            urlLink = "http://feeds.bbci.co.uk/news/technology/rss.xml"
            BBC_News_Feed_Name ="Technology"
    elif chossenFeed =="10":
            urlLink = "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"
            BBC_News_Feed_Name ="EntertainmentArts"
    return (urlLink, BBC_News_Feed_Name)




def readingXMLFile(file,urlLink,BBC_News_Feed_Name):
    #convert to string:
    data = file.read()
    #close file because we don't need it anymore:
    file.close()
      
    #parse the xml you downloaded
    dom = parseString(data)
    
    #-------------- 
    
    
    from xml.dom import minidom
    # To count the number of 
    xmldoc = minidom.parse(urllib2.urlopen(urlLink))
    itemlist = xmldoc.getElementsByTagName('item') 
    
    numOfINews = len(itemlist)
    print " \n\n Number of News Feeds in [",BBC_News_Feed_Name,"] :",numOfINews
    
    
    
    #------------------ Using numOfINews+1 coz of the page title so, by sing +1 will be ignored
    for i in range (0,numOfINews+1):
    
    
        #retrieve the first xml tag (<tag>data</tag>) that the parser finds with name title:
        xmlTagTitle = dom.getElementsByTagName('title')[i+1].toxml()
        #strip off the tag (<title>data</title>  --->   data) 
        xmlDataTitle = xmlTagTitle.replace('<title>','').replace('</title>','').encode('utf-8').strip()
        
        
        xmlTagDescription = dom.getElementsByTagName('description')[i].toxml()
    # xmlDataDescription = xmlTagDescription.replace('<description>','').replace('</description>','')   
    # ---- To Solve    #--------  UnicodeEncodeError: 'ascii' codec can't encode character u'\xa3'
        xmlDataDescription = xmlTagDescription.replace('<description>','').replace('</description>','').encode('utf-8').strip()
    
        
        xmlTagLink = dom.getElementsByTagName('link')[i+1].toxml()
        #strip off the tag (<link>data</link>  --->   data):
        xmlDataLink = xmlTagLink.replace('<link>','').replace('</link>','').encode('utf-8').strip()
        
        if i == 0:
                print "-----------------------------------------------------------------------------------------------"
                print "------------------------------------ [", xmlDataTitle,"] -------------------------------------"
                print "-----------------------------------------------------------------------------------------------"
        
        else:    
            #print out the xml tag and data in this format: <tag>data</tag>
            print "Title :", i, xmlDataTitle
            
             #print out the xml tag and data in this format: <tag>data</tag>
            print "Description :", xmlDataDescription
                 #print out the xml Link and data in this format: <tag>data</tag>
            print "Link :", xmlDataLink,"\n"
            
            # ------  insert values from xml file to RssFeeds Table   
            insertValuesInDB(BBC_News_Feed_Name, xmlDataTitle,xmlDataDescription,xmlDataLink)
        
        
        
              
    return (xmlDataTitle,xmlDataDescription )
