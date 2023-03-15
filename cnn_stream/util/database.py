"""
Author: Foivos Tsimpourlas

Database for processed images stored in http server.
"""
import datetime
import sqlite3
import numpy as np
from typing import Any, Dict, List
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from cnn_stream.util import sqlutil

Base = declarative.declarative_base()

class Image(Base, sqlutil.ProtoBackedMixin):
  """Table for processed images."""
  __tablename__ = "images"
  # entry id
  id: int = sql.Column(sql.Integer, primary_key = True)
  # Image data.
  image: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArray(cls, image: np.array) -> Dict[str, Any]:
    return {
      "image": image,
      "date_added": datetime.datetime.utcnow(),
    }

class ImageDatabase(sqlutil.Database):
  """A database of Images."""

  def __init__(self, url: str, must_exist: bool = False, is_replica: bool = False):
    super(ImageDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self) -> int:
    """Number of samples in DB."""
    with self.Session() as s:
      count = s.query(Sample).count()
    return count

  @property
  def get_data(self) -> List[Image]:
    """Get all DB entries."""
    with self.Session() as s:
      return s.query(Image).yield_per(100)
