    # ------------------ Database Connection ------
import MySQLdb
def connectDB():
        
        # Open database connection
    db = MySQLdb.connect("localhost","root","","TESTDB" )
        
        # prepare a cursor object using cursor() method
    cursor = db.cursor()
        
        # execute SQL query using execute() method.
    #cursor.execute("SELECT VERSION()")
        
        # Fetch a single row using fetchone() method.
    data = cursor.fetchone()
        
    print "Database version : %s " % data
        
        # disconnect from server
    db.close()
    
    return

def createTable():

    
    # Open database connection
    db = MySQLdb.connect("localhost","root","","TESTDB" )
    
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # Drop table if it already exist using execute() method.
    cursor.execute("DROP TABLE IF EXISTS RSSFEEDS")
    
    # Create table as per requirement
    sql = """CREATE TABLE RSSFEEDS (
             FEED_CATEGORY  CHAR(100) NOT NULL,
             TAG_TITLE  CHAR(200) NOT NULL,
             DESCRIPTION  CHAR(200),
             URL_LINK VARCHAR(250)
              )"""
    
    cursor.execute(sql)
    
    # disconnect from server
    db.close()
    return

def insertValuesInDB(BBC_News_Feed_Name, title, xmlDataDescription, urlLink):
     
    print "======================="

    try:
        # Open database connection
        db = MySQLdb.connect("localhost","root","","TESTDB" )

    except :
          print "insertValuesInDB Function Error: unable to connect MySQLdb"

    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # Prepare SQL query to INSERT a record into the database.
   
    
     #sql = """INSERT INTO RSSFEEDS(TAG_TITLE,
     #       DESCRIPTION, URL_LINK)
     #        VALUES ('Macaa', 'Mohan','http://www.bbc.co.uk/news/magazine-26519670#sa-ns_mchannel=rss&amp;ns_source=PublicRSS20-sa ')"""
    
    
    try:
       # Execute the SQL command
       # cursor.execute(sql)   
       cursor.execute('insert into RSSFEEDS values("%s", "%s", "%s", "%s")' % \
            (BBC_News_Feed_Name,title,xmlDataDescription,urlLink ))    
       
       # Commit your changes in the database
       db.commit()
    except e:
       print "--------------------------";
       # Rollback in case there is any error
       db.rollback()
    
    # disconnect from server
    db.close()
    return

def fetchFromServer():
    import MySQLdb
    
    # Open database connection
    db = MySQLdb.connect("localhost","root","","TESTDB" )
    
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    
    # Prepare SQL query to INSERT a record into the database.
    sql = "SELECT * FROM RSSFEEDS \
           WHERE 1 LIMIT 0,3 "
 
   
#    sql = "SELECT * FROM RSSFEEDS \
#          WHERE FEED_CATEGORY = 'Politics' "
    
    
    try:
       # Execute the SQL command
       cursor.execute(sql)
       # Fetch all the rows in a list of lists.
       results = cursor.fetchall()
       for row in results:
          FEED_CATEGORY = row[0]
          TAG_TITLE = row[1]
          DESCRIPTION = row[2]
          URL_LINK = row[3]
    
          
          
          # Now print fetched result
          print "--------------------------------------------------------------------------------------------------------------------------------------"
          print " FEED_CATEGORY : ", FEED_CATEGORY  
          print " TAG_TITLE : ", TAG_TITLE  
          print " DESCRIPTION : ", DESCRIPTION  
          print " URL_LINK : ", URL_LINK  
    
          #print "----------------------------------------------------------"
          #print "fname=%s,lname=%s,age=%s,sex=%s" % \
          #       (FEED_CATEGORY, TAG_TITLE, DESCRIPTION, URL_LINK )
          #print "----------------------------------------------------------"
    
    except:
       print "Error: unable to fecth data"
    
    # disconnect from server
    db.close()

