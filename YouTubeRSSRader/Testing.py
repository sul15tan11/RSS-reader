#import library to do http requests:
import urllib2
import urllib
#checking the SQL #checking the SQL #checking the SQL #checking the SQL 


#import MySQLdb
 
#import easy to use xml parser called minidom:
from xml.dom.minidom import parseString
from urllib2 import URLError 
from xml.dom import minidom


#------------------------------------------------------------------------------------------------

# xmldoc = minidom.parse(urllib2.urlopen('http://feeds.bbci.co.uk/news/rss.xml'))
xmldoc = minidom.parse('books.xml')
itemlist = xmldoc.getElementsByTagName('title') 
print len(itemlist)
    
print itemlist[0].firstChild.nodeValue

for s in itemlist :
    print "Text : ", s.firstChild.nodeValue
    
    
    
#------------------------------------------------------------------------------------------------


       # xmldoc = minidom.parse(urllib2.urlopen('http://feeds.bbci.co.uk/news/rss.xml'))
xmldoc = minidom.parse('items.xml')
itemlist = xmldoc.getElementsByTagName('item') 
print len(itemlist)
print itemlist[0].attributes['name'].value

for s in itemlist :
    print s.attributes['name'].value
    



#---------------------------------------------------------------------------------------------------

urlLink = "http://gdata.youtube.com/feeds/api/videos/iZ_wNWSV-aQ"
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

print "Tiltel :", xmlDataTitle    


# get video description 
xmlTagTitle = xmldoc.getElementsByTagName('content')[0]
print "About :", xmlTagTitle.firstChild.nodeValue


#----------------------------------- different ways
xmlTagTitle = xmldoc.getElementsByTagName('title')[0]
print "Tiltel :", xmlTagTitle.firstChild.nodeValue
        
xmlTagTitle = dom.getElementsByTagName('title')[0]
print "Tiltel :", xmlTagTitle.firstChild.nodeValue

# gives error
""" 
xmlTagTitle = dom.getElementsByTagName('title')[0].toxml()
print "Tiltel :", xmlTagTitle.firstChild.nodeValue
"""        