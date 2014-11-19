#import library to do http requests:
import urllib2
import urllib
#checking the SQL #checking the SQL #checking the SQL #checking the SQL 


#import MySQLdb
 
#import easy to use xml parser called minidom:
from xml.dom.minidom import parseString
from urllib2 import URLError
from xml.dom import minidom

#-----------------------------------------------------------------------------------------------------------
#--------------------------------------------- FUNCTION TO PRINT MENU ----------------
def printMenue():
   # Add both the parameters and return them."

    print " \n "
    print " ---------------------------------  "
    print " |  1- Most recent                | "
    print " |  2- Most viewed                | "
    print " |  3- Top rated                  | "
    print " |  4- Most discussed             | "
    print " |  5- Top favorites              | "
    print " |  6- Most linked                | "
    print " |  7- Recently featured          | "
    print " |  8- Most responded             | "
    print " ---------------------------------  "
    return 

def readingXMLFile(file,urlLink):
    
    #convert to string:
    data = file.read()
    #close file because we don't need it anymore:
    file.close()
      
    #parse the xml you downloaded
    dom = parseString(data)
    
    #-------------- 
    print urlLink
    
    from xml.dom import minidom
    # To count the number of 
    xmldoc = minidom.parse(urllib2.urlopen(urlLink))
    itemlist = xmldoc.getElementsByTagName('item') 
    numOfINews = len(itemlist)
    print " \n\n Number of News Feeds in [ ",numOfINews
    
    
    # this variable to syncronise the category since there are two category in each item
    # so it skips the category which has a link instead of category name
    catSync =1
    pubDateSync=1
    #------------------------------
    
    
    #------------------ Using numOfINews+1 coz of the page title so, by sing +1 will be ignored
    for i in range (0,numOfINews+1):
    
    
        #retrieve the first xml tag (<tag>data</tag>) that the parser finds with name title:
        #using [i+1] to overcome (0,1) coz the first two title are for "most popular"
        
        # it can be used ( if i ==0 or 1) and else will start with [2] instead of [i+1]
        xmlTagTitle = dom.getElementsByTagName('title')[i+1].toxml()
        #strip off the tag (<title>data</title>  --->   data) 
        xmlDataTitle = xmlTagTitle.replace('<title>','').replace('</title>','').encode('utf-8').strip()
        
        xmlTagLink = dom.getElementsByTagName('link')[i+1].toxml()
        #strip off the tag (<link>data</link>  --->   data):
        xmlDataLink = xmlTagLink.replace('<link>','').replace('&amp;feature=youtube_gdata</link>','').encode('utf-8').strip()
        xmlVidID = xmlTagLink.replace('<link>http://www.youtube.com/watch?v=','').replace('&amp;feature=youtube_gdata</link>','').encode('utf-8').strip()
        
        

        
        
        xmlTagcategory = dom.getElementsByTagName('category')[i+catSync]
    # xmlDataDescription = xmlTagDescription.replace('<description>','').replace('</description>','')   
    # ---- To Solve    #--------  UnicodeEncodeError: 'ascii' codec can't encode character u'\xa3'
       # xmlDatacategory = xmlTagcategory.replace('<category>','').replace('</category>','').encode('utf-8').strip()
   
   
   
        if i == 0:
                print "-----------------------------------------------------------------------------------------------"
                print "------------------------------------- [", xmlDataTitle,"] --------------------------------------"
                print "-----------------------------------------------------------------------------------------------"
        
        else:    
            #print out the xml tag and data in this format: <tag>data</tag>
            print "Tiltel :", i, xmlDataTitle
            print "xmlVidID :", xmlVidID
            print "Link xml: http://gdata.youtube.com/feeds/api/videos/"+xmlVidID
            print "Link :", xmlDataLink
            
            #---- using (i-1) coz the i the loop will be (numOfINews+1) out of range and <pubDate> is less than others tags
            xmlPubDate = dom.getElementsByTagName('pubDate')[i-1].toxml()
            #strip off the tag (<link>data</link>  --->   data):
            xmlPubDate = xmlPubDate.replace('<pubDate>','').replace('</pubDate>','').encode('utf-8').strip()
            print "Publishing date :", xmlPubDate
            
            xmlPubDate = dom.getElementsByTagName('pubDate')[i-1]
            print "Publishing date:", xmlPubDate.firstChild.nodeValue 
            
            #print "Description :", xmlDatacategory
            print "category :", xmlTagcategory.firstChild.nodeValue
            print "\n"
            catSync+=1;
            
    getVidDetails(xmlVidID)    

    return xmlDataTitle
# --------------------------------------------------------------------------

def getVidDetails(xmlVidID):
    
        
    urlLink = "http://gdata.youtube.com/feeds/api/videos/"+xmlVidID
   
    file = urllib2.urlopen(urlLink)
    
    
    data = file.read()
        #close file because we don't need it anymore:
    file.close()
          
        #parse the xml you downloaded
    dom = parseString(data)
    
    
    #######
    ## 1 ## use ( dom ) dom.getElementsByTagName('title')[0].toxml() or to use replace fun need .toxml() coz u ll not get the value
    #######
    
    
    xmldoc = minidom.parse(urllib2.urlopen(urlLink))
    
    xmlTagTitle = dom.getElementsByTagName('title')[0].toxml()
            #strip off the tag (<title>data</title>  --->   data) 
    xmlDataTitle = xmlTagTitle.replace('<title type="text">','').replace('</title>','').encode('utf-8').strip()
    
    vidURL= "http://www.youtube.com/watch?v="+xmlVidID 
    vidCommentsURL = "http://gdata.youtube.com/feeds/api/videos/"+xmlVidID+"/comments" 

    print "Title:", xmlDataTitle   
    print "Video ID:", xmlVidID 
    print "Video URL:", vidURL 
    print "Video Comments URL:", vidCommentsURL 
    
    itemlist = xmldoc.getElementsByTagName('category') 
    print "category:", itemlist[1].attributes['label'].value
    
    xmlPubDate = dom.getElementsByTagName('published')[0]
    print "Publishing date:", xmlPubDate.firstChild.nodeValue 
    
    # get video description 
    xmlTagAbout = xmldoc.getElementsByTagName('content')[0]
    print "About :", xmlTagAbout.firstChild.nodeValue


 
    return 

def getNewsFeedsURL( chossenFeed ):
    
    urlLink = ""
    BBC_News_Feed_Name =""
    
    if chossenFeed == "1":
            urlLink = "http://feeds.bbci.co.uk/news/rss.xml"
            BBC_News_Feed_Name ="Top Stories"
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
            urlLink = "http://feeds.bbci.co.uk/news/politics/rss.xml"
            BBC_News_Feed_Name ="Health"
    elif chossenFeed =="7":
            urlLink = "http://feeds.bbci.co.uk/news/education/rss.xml"
            BBC_News_Feed_Name ="Education & Family"
    elif chossenFeed =="8":
            urlLink = "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"
            BBC_News_Feed_Name ="Science & Environment"
    elif chossenFeed =="9":
            urlLink = "http://feeds.bbci.co.uk/news/technology/rss.xml"
            BBC_News_Feed_Name ="Technology"
    elif chossenFeed =="10":
            urlLink = "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"
            BBC_News_Feed_Name ="Entertainment & Arts"
    return (urlLink, BBC_News_Feed_Name)



#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------




# -------