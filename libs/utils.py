from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from mapping import crime_cat_code_map 
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

def start_spark():
    try:
        spark = SparkSession.builder.appName("CrimeDataProcessing").getOrCreate()
    except Exception as e:
        print(f"Error initializing Spark session: {e}")
    
    return spark

def drop_duplicates(df):
    df.dropDuplicates()
    return df
    
def check_missing_values(df):
    missing_df = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    return missing_df

def handle_missing_values(df):
    # replace the values ​​below 1 for the "Vict Age" column with NaN
    df = df.withColumn("Vict Age", F.when(F.col("Vict Age") < 1, None).otherwise(F.col("Vict Age")))
    df = df.dropna(subset=["Vict Age"])
    
    # fill missing values in 'Vict Sex' with 'X'
    df = df.fillna({'Vict Sex': 'X'})
    # there are some entries in the 'Vict Sex' with values 'H' and '-'
    # changing them to 'X' since it's not explained in the notes what are they
    df = df.withColumn("Vict Sex", F.when(df["Vict Sex"] == "H", "X").otherwise(df["Vict Sex"]))
    df = df.withColumn("Vict Sex", F.when(df["Vict Sex"] == "-", "X").otherwise(df["Vict Sex"]))
    
    # fill missing values in categorical columns with a string such as "N/A"
    categorical_cols = [c for c in df.columns if dict(df.dtypes)[c] == 'object']
    
    for column in categorical_cols:
        if df.filter(F.col(column).isNull()).count() > 0:
            df = df.withColumn(column, F.when(F.col(column).isNull(), 'N/A').otherwise(F.col(column)))

    # delete rows with empty 'Crm Cd', 'Status' or empty detailed location
    df.na.drop(subset = ['Crm Cd', 'Status', 'LAT', 'LON'])
    
    # clean out those lat long with 0.0 values
    df = df[(df["LAT"]!=0.0)&(df["LON"]!=0.0)]

    return df

def read_ori_data(spark, file_path):
    df = spark.read.option("header", "true").csv(file_path, inferSchema = True)
    return df

def prepare_data(spark, file_path):
    df = spark.read.option("header", "true").csv(file_path, inferSchema = True)
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    return df


def feature_selection(df):
    # delete columns with redundant information
    df = df.drop('DR_NO', 'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'Cross Street')
    return df

def assign_time_slot(h):
    if h < 8:
        return 1 # time slot from 0:00 - 7:59
    elif h < 16:
        return 2 # time slot from 8:00 - 15:59
    else:
        return 3 # time slot from 16:00 - 23:59
    
assign_time_slot_udf = udf(assign_time_slot, IntegerType())

def convert_datetime(df):
    # convert 'Date Rptd' and 'DATE OCC' into datetime
    df = df.withColumn("Date Rptd", F.to_date(F.col("Date Rptd"), "MM/dd/yyyy hh:mm:ss a"))
    df = df.withColumn("DATE OCC", F.to_date(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"))

    # convert 'TIME OCC' into TimeType
    # TODO: the formatting here is slightly wrong. need to check again.
    df = df.withColumn("TIME OCC", F.lpad(F.col("TIME OCC").cast("string"), 4, '0'))  # Pad with zeros to ensure length of 4
    df = df.withColumn("TIME OCC", F.concat(F.substring("TIME OCC", 1, 2), F.lit(':'), F.substring("TIME OCC", 3, 2)))  # Format as HH:mm
    
    df = df.withColumn("Occ DateTime", F.to_timestamp(F.concat_ws(" ", df["DATE OCC"], df["TIME OCC"])))
    df = df.withColumn("hour", F.hour(df["Occ DateTime"]))
    df = df.withColumn("time_slot", assign_time_slot_udf(F.col("hour")))

    # extract year, month and day of the week from datetime
    df = df.withColumn('Year OCC', F.year("DATE OCC"))
    df = df.withColumn('Month OCC', F.month("DATE OCC"))
    df = df.withColumn('Day OCC', F.dayofweek(F.col("DATE OCC")))
    df = df.withColumn('Date OCC', F.day("DATE OCC"))
    return df
    

def vict_cd_transform(df):
    # Mapping of victim codes to descriptions
    df = df.withColumn(
        "Vict Descent",
        F.when(F.col("Vict Descent") == "A", "Other Asian")
        .when(F.col("Vict Descent") == "B", "Black")
        .when(F.col("Vict Descent") == "C", "Chinese")
        .when(F.col("Vict Descent") == "D", "Cambodian")
        .when(F.col("Vict Descent") == "F", "Filipino")
        .when(F.col("Vict Descent") == "G", "Guamanian")
        .when(F.col("Vict Descent") == "H", "Hispanic/Latin/Mexican")
        .when(F.col("Vict Descent") == "I", "American Indian/Alaskan Native")
        .when(F.col("Vict Descent") == "J", "Japanese")
        .when(F.col("Vict Descent") == "K", "Korean")
        .when(F.col("Vict Descent") == "L", "Laotian")
        .when(F.col("Vict Descent") == "O", "Other")
        .when(F.col("Vict Descent") == "P", "Pacific Islander")
        .when(F.col("Vict Descent") == "S", "Samoan")
        .when(F.col("Vict Descent") == "U", "Hawaiian")
        .when(F.col("Vict Descent") == "V", "Vietnamese")
        .when(F.col("Vict Descent") == "W", "White")
        .when(F.col("Vict Descent") == "X", "Unknown")
        .when(F.col("Vict Descent") == "Z", "Asian Indian")
        .otherwise("Unknown"))
    return df
    
def add_age_group(df):
    # Add the 'age_group' column
    df = df.withColumn("age_group", 
        F.when((F.col("Vict Age") <1), "Infant")
        .when((F.col("Vict Age") >= 1) & (F.col("Vict Age") <= 4), "Toddler")
        .when((F.col("Vict Age") >= 5) & (F.col("Vict Age") <= 12), "Kid")
        .when((F.col("Vict Age") >= 13) & (F.col("Vict Age") <= 17), "Teenager")
        .when((F.col("Vict Age") >= 18) & (F.col("Vict Age") <= 25), "Young Adult")
        .when((F.col("Vict Age") >= 26) & (F.col("Vict Age") <= 39), "Adult")
        .when((F.col("Vict Age") >= 40) & (F.col("Vict Age") <= 65), "Middle Age")
        .when((F.col("Vict Age") > 65), "Senior")
        .otherwise("Unknown"))
    return df

def map_crime_cd_to_cat(crm_cd):
    crm_cd = str(crm_cd)
    return crime_cat_code_map.get(crm_cd, "Unknown")

def map_crime_cat(df):
    map_udf = F.udf(map_crime_cd_to_cat, StringType())

    df = df.withColumn("Category", map_udf(df["Crm Cd"]))
    return df

def feature_engineering(df):
    df = feature_selection(df)
    df = convert_datetime(df)
    df = vict_cd_transform(df)
    df = add_age_group(df)
    df = map_crime_cat(df)
    
    return df
    

def drop_columns(df, column_list):
    df = df.drop(*column_list)
    return df

def convert_to_timestamp(df, columnname):
    df = df.withColumn("datetime_str", F.to_timestamp(columnname, "yyyy-MM-dd HH:mm"))
    return df
def convert_timestamp(df, columnname):
    df = df.withColumn(columnname, F.to_timestamp(columnname, "yyyy-MM-dd"))
    return df
    
    
def concat_datetime_columns(df, date_col, time_col):
    df = df.withColumn("datetime_occ", F.concat(F.col(date_col), F.lit(" "), F.col(time_col)))
    df = convert_to_timestamp(df, "datetime_occ")
    return df

def ohe_all_cats(df, cat_cols, num_cols):
    indexers = []
    encoders = []
    for col in cat_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        encoder = OneHotEncoder(inputCols=[f"{col}_index"], outputCols=[f"{col}_ohe"])
        indexers.append(indexer)
        encoders.append(encoder)
        
    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df)
    df_encoded = model.transform(df)
    ohe_columns = [f"{col}_ohe" for col in cat_cols]
    assembler = VectorAssembler(inputCols=ohe_columns + num_cols, outputCol="features")
    df_features = assembler.transform(df_encoded)
    return df_features

def ohe_cat(df, categorical_cols):
    indexers = []
    encoders = []

    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        encoder = OneHotEncoder(inputCols=[f"{col}_index"], outputCols=[f"{col}_ohe"])
        indexers.append(indexer)
        encoders.append(encoder)

    # Create a Pipeline for One-Hot Encoding
    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df)
    df_encoded = model.transform(df)

    # Step 2: Assemble One-Hot Encoded Vectors
    # Collect all one-hot encoded columns
    ohe_columns = [f"{col}_ohe" for col in categorical_cols]

    # Use VectorAssembler to combine them into a single feature vector
    assembler = VectorAssembler(inputCols=ohe_columns, outputCol="features")
    df_features = assembler.transform(df_encoded)
    return df_features


def stop_spark(spark):
    spark.stop() 