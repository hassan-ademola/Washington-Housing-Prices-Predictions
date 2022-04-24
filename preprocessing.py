from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from top_features import top_feats
from all_features import all_feats
import numpy as np
import pandas as pd


class Add_attr(BaseEstimator,TransformerMixin):
    
    regions = {'80':'1','81':'2','82':'3','83':'4'}
    added_cols = ['has_basement','was_renovated','region','quadrant','road_type',
                  'total_rooms','livarea/bedroom','lotarea/bedroom','livarea/bathroom',
                  'lotarea/bathroom','land_space','total_space','landspace/room',
                  'totspace/room','facilities','score']

    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        df = X.copy()
        df['has_basement'] = (df.sqft_basement!=0).astype(int)
        df['was_renovated'] = (df.yr_renovated!=0).astype(int)
        codes = df.zip.apply(lambda x: x[1:3])
        df['region'] = codes.map(self.regions)
        df['quadrant'] = df.street.apply(lambda x: 'NW' if 'NW' in x.split()
                                else 'NE' if 'NE' in x.split()
                                else 'SE' if 'SE' in x.split()
                                else 'SW' if 'SW' in x.split()
                                else 'UNKNOWN')
        df['road_type'] = df.street.apply(lambda x: 'Rd' if ('Rd' in x.split()) or ('Road' in x.split())
                            else 'Way' if 'Way' in x.split()
                            else 'St' if ('St' in x.split()) or ('Street' in x.split())
                            else 'Ave' if ('Ave' in x.split()) or ('Avenue' in x.split())
                            else 'Blvd' if ('Blvd' in x.split()) or ('Boulevard' in x.split())
                            else 'Ln' if ('Ln' in x.split()) or ('Lane' in x.split())
                            else 'Dr' if ('Dr' in x.split()) or ('Drive' in x.split())
                            else 'Pl' if ('Pl' in x.split()) or ('Place' in x.split())
                            else 'Ct' if ('Ct' in x.split()) or ('Court' in x.split())
                            else 'Pkwy' if ('Pkwy' in x.split()) or ('Parkway' in x.split())
                            else 'Hwy' if 'Hwy' in x.split()
                            else 'Terrace' if 'Terrace' in x.split()
                            else 'Trail' if 'Trail' in x.split()
                            else 'Cir' if 'Cir' in x.split()
                            else 'Walk' if 'Walk' in x.split()
                            else 'Key' if 'Key' in x.split()
                            else 'UNKNOWN')
        df['total_rooms'] = df['bedrooms']+df['bathrooms']
        df['livarea/bedroom'] = df['sqft_living']/df['bedrooms']
        df['lotarea/bedroom'] = df['sqft_lot']/df['bedrooms']
        df['livarea/bathroom'] = df['sqft_living']/df['bathrooms']
        df['lotarea/bathroom'] = df['sqft_lot']/df['bathrooms']
        df['land_space'] = df['sqft_living']+df['sqft_lot']
        df['total_space'] = df['sqft_living']+df['sqft_lot']+df['sqft_above']+df['sqft_basement']
        df['landspace/room'] = df['land_space']/df['total_rooms']
        df['totspace/room'] = df['total_space']/df['total_rooms']
        df['facilities'] = df.apply(lambda x:int(x.bedrooms!=0)+int(x.bathrooms!=0)+int(x.view!=0)
                                    +int(x.waterfront!=0)+int(x.sqft_basement!=0),axis=1)
        df['score'] = df['facilities']*df['condition']
        
        # some division operations might return infinity values
        df = df.replace(np.inf,dict(df.median(numeric_only=True)))
        return df

    
class Encode(BaseEstimator,TransformerMixin):
    ordinal = ['waterfront','has_basement','was_renovated']
    target_enc = ['city','zip','road_type']
    nominal = ['view','condition','region','quadrant']

    def __init__(self,m=1,n=1):
        self.hot_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
        self.m = m
        self.n = n
        self.sqft_encoder = TargetEncoder(smoothing=m)
        self.lot_encoder = TargetEncoder(smoothing=n)
    
    def fit(self,X,y=None):
        self.hot_encoder.fit(X[self.nominal])
        X_enc = X[['city','zip','road_type']]
        y_sqft = X.sqft_living
        y_lot = X.sqft_lot
        self.sqft_encoder.fit(X_enc,y_sqft)
        self.lot_encoder.fit(X_enc,y_lot)
        return self
       
    
    def transform(self,X,y=None):
        df = X.reset_index(drop=True)
        dummy_data = self.hot_encoder.transform(df[self.nominal])
        self.dummy_cols = self.hot_encoder.get_feature_names_out(self.nominal)
        dummy_df = pd.DataFrame(dummy_data,columns=self.dummy_cols)
        ord_df = df[self.ordinal].reset_index(drop=True)
        cat_df = pd.concat([ord_df,dummy_df],axis=1)
        cat_df[['sqft_city','sqft_zip','sqft_roadtype']] = self.sqft_encoder.transform(df[['city','zip',
                                                                                          'road_type']])
        cat_df[['lot_city','lot_zip','lot_roadtype']] = self.lot_encoder.transform(df[['city','zip',
                                                                                       'road_type']])
        self.cat_cols = cat_df.columns
        return cat_df

    
class Pass(BaseEstimator,TransformerMixin):
    num_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above',
                'sqft_basement','yr_built','yr_renovated','total_rooms','livarea/bedroom',
                'lotarea/bedroom','livarea/bathroom','lotarea/bathroom','land_space','total_space',
                'landspace/room','totspace/room','facilities','score']
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        return X[self.num_cols]

    
class Select(BaseEstimator,TransformerMixin):
    def __init__(self,n=30):
        self.n=n
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):   
        # get the indices of the top n features
        self.indices = []
        for feat in top_feats[:self.n]:
            self.indices.append(all_feats.index(feat))
        return X[:,self.indices]