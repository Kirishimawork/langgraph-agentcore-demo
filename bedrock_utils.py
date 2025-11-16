import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# Global Bedrock client
REGION_NAME = os.getenv('AWS_DEFAULT_REGION')
BEDROCK = boto3.client('bedrock-runtime', region_name=REGION_NAME)


def _call_bedrock(payload: dict, modelId, encode: bool = False) -> dict:
    """Internal helper to invoke Bedrock and return parsed JSON response."""
    body = json.dumps(payload).encode('utf-8') if encode else json.dumps(payload)
    resp = BEDROCK.invoke_model(
        body=body,
        modelId=modelId,
        accept='application/json',
        contentType='application/json',
    )
    return json.loads(resp['body'].read())


def query_llm(prompts: str, modelId):
    """แปลงภาษามนุษย์ -> SQL (คืนข้อความผลลัพธ์ตรงๆ)

    ห้ามเปลี่ยน logic: จะคืนค่า model_response["output"]["message"]["content"][0]["text"]
    """
    payload = {
        'messages': [
            {'role': 'user', 'content': [{'text': prompts}]}
        ]
    }

    model_response = _call_bedrock(payload, modelId, encode=False)
    # Nova Pro ตอบใน outputs[0].content[0]["text"] ตามโค้ดเดิม
    return model_response['output']['message']['content'][0]['text']


def qna_llm(prompts: str, modelId):
    """แปลงผลลัพธ์ SQL (ตาราง) -> ภาษามนุษย์

    คืนข้อความเหมือนเดิม แต่มีการจัดการกรณีรูปแบบตอบไม่คาดคิด
    """
    payload = {'messages': [{'role': 'user', 'content': [{'text': prompts}]}]}

    model_response = _call_bedrock(payload, modelId, encode=True)
    try:
        return model_response['output']['message']['content'][0]['text']
    except (KeyError, IndexError):
        return f"Unexpected response format: {model_response}"


def llm_debugger(statement: str, error: str, params: dict) -> str:
    """สร้าง prompt สำหรับดีบักข้อผิดพลาด SQL (คืนเฉพาะ SQL ที่แก้แล้ว)

    หลีกเลี่ยงการเปลี่ยน logic: ใช้ query_llm(prompts, modelId=None) เหมือนเดิม
    """
    prompts = rf'''<s><<SYS>>[INST]
You are a PostgreSQL developer who is an expert at debugging errors.

Here are the schema definition of table(s):
{params['schema']}
#############################
Here are example records for each table:
{params['sample']}
#############################
Here is the sql statement that threw the error below:
{statement}
#############################
Here is the error to debug:
{error}
#############################
Here is the intent of the user:
{params['prompt']}
<</SYS>>
First understand the error and think about how you can fix the error.
Use the provided schema and sample row to guide your thought process for a solution.
Do all this thinking inside <thinking></thinking> XML tags.This is a space for you to write down relevant content and will not be shown to the user.

Once your are done debugging, provide the the correct SQL statement without any additional text.
When generating the correct SQL statement:
1. Pay attention to the schema and table name and use them correctly in your generated sql.
2. Never query for all columns from a table unless the question says so. You must query only the columns that are needed to answer the question.
3. Wrap each column name in double quotes (") to denote them as delimited identifiers. Do not use backslash (\\) to escape underscores (_) in column names.

Format your response as:
<sql> Correct SQL Statement </sql>[/INST]'''

    answer = query_llm(prompts, modelId=None)
    return answer.replace('\\', '')
