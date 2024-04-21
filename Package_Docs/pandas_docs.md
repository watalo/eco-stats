# pandas文档

代码实现中用到的内容。
## C1.数据读取

`pandas.read_stata()`可直接读取`*.dta`数据。
- 可选参数：
	- _filepath_or_buffer_, 字符串形式文件路径，URL等
	- _*_, 
	- _convert_dates=True_, 是否转换为pandas时间格式，默认True
	- _convert_categoricals=True_, 
	- _index_col=None_, 
	- _convert_missing=False_, 
	- _preserve_dtypes=True_, 
	- _columns=None_, 
	- _order_categoricals=True_, 
	- _chunksize=None_, 
	- _iterator=False_, 
	- _compression='infer'_, 
	- _storage_options=Non_

## C2.时间序列处理方法

Pandas 是一个强大的 Python 数据分析工具库，它提供了丰富的功能来处理时间序列数据。以下是一些主要的 pandas 时间序列处理方法，以及相应的代码片段说明：
### 1. 解析时间序列信息：
使用 `pd.to_datetime()` 函数可以将各种格式的日期时间字符串转换为 pandas 的 `Timestamp` 对象或 `DatetimeIndex`。

```python
dti = pd.to_datetime(["1/1/2018", "2018-01-01", "2018-01-01 00:00:00"])
```
### 2. 生成固定频率的日期序列：
`pd.date_range()` 函数可以生成一个固定频率的日期时间序列。

```python
dti = pd.date_range("2018-01-01", periods=3, freq="h")
```
### 3. 时区处理：
通过 `tz_localize()` 和 `tz_convert()` 方法，可以指定或转换时间序列的时区。

```python
dti = dti.tz_localize("UTC")
dti = dti.tz_convert("US/Pacific")
```
### 4. 重采样（Resampling）：
 `resample()` 方法允许用户根据时间频率对数据进行上采样或下采样，并应用聚合函数。

```python
ts = pd.Series(range(len(idx)), index=idx)
ts.resample("2h").mean()
```
### 5. 时间算术：
 执行绝对或相对时间增量的日期和时间算术。

```python
friday = pd.Timestamp("2018-01-05")
saturday = friday + pd.Timedelta("1 day")
```
### 6. 时间序列表示：
 pandas 可以表示四种一般的时间相关概念：日期时间、时间增量、时间跨度和日期偏移。

```python
DatetimeIndex, TimedeltaIndex, PeriodIndex, DateOffset
   ```
### 7. 转换为时间戳：
将类似对象的 Series 或列表转换为 `DatetimeIndex`。

   ```python
pd.to_datetime(["2005/11/23", "2010/12/31"])
   ```
### 8. 处理不匹配的数据：
当遇到无法解析的日期时间字符串时，可以通过 `errors` 参数指定如何处理。

   ```python
pd.to_datetime(["2009/07/31", "asd"], errors="coerce")
   ```
### 9. 生成时间戳范围：
使用 `date_range()` 或 `bdate_range()` 可以生成一系列时间戳。

   ```python
stamps = pd.date_range("2012-10-08 18:15:05", periods=4, freq="D")
   ```
### 10. 自定义频率范围：

使用 `bdate_range()` 可以生成具有自定义频率的日期范围。

```python
pd.bdate_range(end=end, periods=20)
```

### 11. 时间戳限制：
了解时间戳表示的限制，例如纳秒分辨率的时间跨度限制。
```python
pd.Timestamp.min, pd.Timestamp.max
```
### 12. 索引和切片：
`DatetimeIndex` 可以作为 pandas 对象的索引，并提供了智能的索引功能。

```python
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts[:5]
```
### 13. 时间/日期组件：
可以直接从 `Timestamp` 或 `DatetimeIndex` 访问时间/日期的组成部分。

```python
ts.index.year, ts.index.month, ts.index.day, etc.
```
### 14. 使用 DateOffset 对象：
`DateOffset` 对象允许执行特定的日期时间偏移。

```python
ts = pd.Timestamp("2016-10-30 00:00:00", tz="Europe/Helsinki")
ts + pd.DateOffset(days=1)
```
### 15. 自定义节假日和假期日历：
使用 `HolidayCalendar` 可以定义自定义节假日。

```python
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar
class ExampleCalendar(AbstractHolidayCalendar):
	rules = [
		Holiday("Columbus Day", month=10, day=1, offset=pd.DateOffset(weekday=MO(2))),
	]
```
### 16. 时间序列相关的实例方法：

如 `shift()` 用于时间序列的移动，`asfreq()` 用于频率转换。

```python
ts.shift(1)
ts.asfreq()
```
### 17. 时间序列重采样：
`resample()` 方法用于改变时间序列的频率，可以进行求和、平均、最大值等聚合操作。

```python
ts.resample("5Min").sum()
```
### 18. 时间跨度表示：
使用 `Period` 和 `PeriodIndex` 来表示时间跨度。

```python
p = pd.Period("2012", freq="Y-DEC")
p + 1  # 转换为 2013
```
### 19. 时区处理：
使用 `tz_localize()` 和 `tz_convert()` 方法处理时区相关的转换。

```python
rng = pd.date_range("3/6/2012 00:00", periods=3, freq="D")
rng_utc = rng.tz_convert("UTC")
```

这些方法为处理时间序列数据提供了强大的工具集，允许用户执行复杂的时间数据分析和操作。