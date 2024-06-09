# Lagent & AgentLego 智能体应用搭建

## Lagent 

### 环境搭建

```bash
# 创建目录
mkdir -p /root/agent

# 安装环境
studio-conda -t agent -o pytorch-2.1.2

# 安装Lagent和AgentLego
cd /root/agent
conda activate agent
git clone https://gitee.com/internlm/lagent.git
cd lagent && git checkout 581d9fb && pip install -e . && cd ..
git clone https://gitee.com/internlm/agentlego.git
cd agentlego && git checkout 7769e0d && pip install -e . && cd ..

# 安装其他依赖
conda activate agent
pip install lmdeploy==0.3.0

# 准备tutorial
cd /root/agent
git clone -b camp2 https://gitee.com/internlm/Tutorial.git
```



### 部署服务端

```bash
lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b --server-name 127.0.0.1 --model-name internlm2-chat-7b --cache-max-entry-count 0.1
```

![image-20240609125603926](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture6/image-20240609125603926.png)



### 部署前端服务

另外打开一个终端

```bash
conda activate agent
cd /root/agent/lagent/examples
streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860
```



### 端口映射

打开本地机器的powershell

```bash
ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的 ssh 端口号
```



### 使用

1. 访问 http://127.0.0.1:7860 
2. 修改模型IP为：127.0.0.1:23333
3. 选择工具



![image-20240609125947008](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture6/image-20240609125947008.png)



### 自定义工具

#### 创建python文件

```bash
touch /root/agent/lagent/lagent/actions/weather.py
```

```python
import json
import os
import requests
from typing import Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

class WeatherQuery(BaseAction):
    """Weather plugin for querying weather information."""
    
    def __init__(self,
                 key: Optional[str] = None,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        key = os.environ.get('WEATHER_API_KEY', key)
        if key is None:
            raise ValueError(
                'Please set Weather API key either in the environment '
                'as WEATHER_API_KEY or pass it as `key`')
        self.key = key
        self.location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
        self.weather_query_url = 'https://devapi.qweather.com/v7/weather/now'

    @tool_api
    def run(self, query: str) -> ActionReturn:
        """一个天气查询API。可以根据城市名查询天气信息。
        
        Args:
            query (:class:`str`): The city name to query.
        """
        tool_return = ActionReturn(type=self.name)
        status_code, response = self._search(query)
        if status_code == -1:
            tool_return.errmsg = response
            tool_return.state = ActionStatusCode.HTTP_ERROR
        elif status_code == 200:
            parsed_res = self._parse_results(response)
            tool_return.result = [dict(type='text', content=str(parsed_res))]
            tool_return.state = ActionStatusCode.SUCCESS
        else:
            tool_return.errmsg = str(status_code)
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return
    
    def _parse_results(self, results: dict) -> str:
        """Parse the weather results from QWeather API.
        
        Args:
            results (dict): The weather content from QWeather API
                in json format.
        
        Returns:
            str: The parsed weather results.
        """
        now = results['now']
        data = [
            f'数据观测时间: {now["obsTime"]}',
            f'温度: {now["temp"]}°C',
            f'体感温度: {now["feelsLike"]}°C',
            f'天气: {now["text"]}',
            f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
            f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
            f'相对湿度: {now["humidity"]}',
            f'当前小时累计降水量: {now["precip"]} mm',
            f'大气压强: {now["pressure"]} 百帕',
            f'能见度: {now["vis"]} km',
        ]
        return '\n'.join(data)

    def _search(self, query: str):
        # get city_code
        try:
            city_code_response = requests.get(
                self.location_query_url,
                params={'key': self.key, 'location': query}
            )
        except Exception as e:
            return -1, str(e)
        if city_code_response.status_code != 200:
            return city_code_response.status_code, city_code_response.json()
        city_code_response = city_code_response.json()
        if len(city_code_response['location']) == 0:
            return -1, '未查询到城市'
        city_code = city_code_response['location'][0]['id']
        # get weather
        try:
            weather_response = requests.get(
                self.weather_query_url,
                params={'key': self.key, 'location': city_code}
            )
        except Exception as e:
            return -1, str(e)
        return weather_response.status_code, weather_response.json()
```



#### 启动前端

在 `https://dev.qweather.com/docs/api/` 获取api key，启动服务端后，启动前端的命令修改如下

```bash
# 设置变量 WEATHER_API_KEY
export WEATHER_API_KEY=XXX
# 比如 export WEATHER_API_KEY=1234567890abcdef
conda activate agent
cd /root/agent/Tutorial/agent
streamlit run internlm2_weather_web_demo.py --server.address 127.0.0.1 --server.port 7860
```



![image-20240609131252255](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture6/image-20240609131252255.png)



## AgentLego

AgentLego是一个算法库，可以直接使用，也可以作为智能体工具使用



### 下载demo图片

```bash
cd /root/agent
wget http://download.openmmlab.com/agentlego/road.jpg
```



### 安装依赖

```bash
conda activate agent
pip install openmim==0.3.9
mim install mmdet==3.3.0
```



### 创建python文件

```bash
touch /root/agent/direct_use.py
```

```python
import re

import cv2
from agentlego.apis import load_tool

# load tool
tool = load_tool('ObjectDetection', device='cuda')

# apply tool
visualization = tool('/root/agent/road.jpg')
print(visualization)

# visualize
image = cv2.imread('/root/agent/road.jpg')

preds = visualization.split('\n')
pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'

for pred in preds:
    name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
    x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
```



### 执行python文件

```bash
python /root/agent/direct_use.py
```



### 查看结果

![image-20240609132948900](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture6/image-20240609132948900.png)



![image-20240609133003568](https://github.com/la-gluha/InternStudio/blob/main/resource/img/lecture6/image-20240609133003568.png)





