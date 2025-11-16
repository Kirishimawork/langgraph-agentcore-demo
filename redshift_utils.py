import os
import time
import boto3
import pandas as pd
from dotenv import load_dotenv
from bedrock_utils import llm_debugger

load_dotenv()

# Global Redshift Data API client
REGION_NAME = os.getenv('AWS_DEFAULT_REGION')
REDSHIFT_DATA = boto3.client('redshift-data', region_name=REGION_NAME)


def get_db_redshift(conn_param, database):
    """ดึงรายการฐานข้อมูลใน Redshift cluster/workgroup"""
    return REDSHIFT_DATA.list_databases(**conn_param, Database=database)['Databases']


def get_schema_redshift(conn_param, database):
    """ดึงรายการ schema จาก Redshift"""
    return REDSHIFT_DATA.list_schemas(**conn_param, Database=database)['Schemas']


def get_tables_redshift(conn_param, database, schema):
    """ดึงชื่อ table ทั้งหมดใน schema ที่ระบุ"""
    tables = REDSHIFT_DATA.list_tables(**conn_param, Database=database, SchemaPattern=schema)
    return [t['name'] for t in tables['Tables']]


def get_redshift_table_result(response):
    """แปลงผลลัพธ์จาก Redshift query เป็น CSV string"""
    cols = [c['name'] for c in response['ColumnMetadata']]
    data = [[list(v.values())[0] for v in r] for r in response['Records']]
    return pd.DataFrame(data, columns=cols).to_csv(index=False)


def execute_query_redshift(sql, conn_param, database):
    """รัน SQL เดียวใน Redshift แล้วคืน response object"""
    return REDSHIFT_DATA.execute_statement(**conn_param, Database=database, Sql=sql)


def _wait_for_statement(statement_id, timeout=300):
    """รอให้ statement ทำงานเสร็จ พร้อมตรวจสอบ error"""
    start = time.time()
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Query timed out after {timeout}s")
        desc = REDSHIFT_DATA.describe_statement(Id=statement_id)
        status = desc['Status']
        if status == 'FINISHED':
            return desc
        if status in ('FAILED', 'ABORTED'):
            raise RuntimeError(desc.get('Error', 'Unknown error'))
        time.sleep(1)


def execute_query_with_pagination(sql_list, conn_param, database, max_wait_seconds=300):
    """รองรับการรัน query หลายตัวและรอผลลัพธ์แบบมี timeout"""
    results = []
    try:
        if 'ClusterIdentifier' in conn_param:
            resp = REDSHIFT_DATA.batch_execute_statement(**conn_param, Database=database, Sqls=sql_list)
            desc = _wait_for_statement(resp['Id'], max_wait_seconds)
            for sub in desc.get('SubStatements', []):
                if sub.get('Status') == 'FAILED':
                    raise RuntimeError(sub.get('Error', 'Unknown error'))
                res = REDSHIFT_DATA.get_statement_result(Id=sub['Id'])
                results.append(get_redshift_table_result(res))

        elif 'WorkgroupName' in conn_param:
            for sql in sql_list:
                resp = execute_query_redshift(sql, conn_param, database)
                _wait_for_statement(resp['Id'], max_wait_seconds)
                res = REDSHIFT_DATA.get_statement_result(Id=resp['Id'])
                results.append(get_redshift_table_result(res))
        else:
            raise ValueError('connection_param ต้องมี ClusterIdentifier หรือ WorkgroupName')
        return results

    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")
        raise


def redshift_querys(q_s, response, params, conn_param, database):
    """รัน query บน Redshift ถ้า error ให้ดีบัก SQL ด้วย llm_debugger แล้วลองใหม่"""
    max_try, debug_count = 5, 5
    try:
        stmt_result = REDSHIFT_DATA.get_statement_result(Id=response['Id'])
    except REDSHIFT_DATA.exceptions.ResourceNotFoundException:
        desc = REDSHIFT_DATA.describe_statement(Id=response['Id'])
        status = desc['Status']
        while status in ('SUBMITTED', 'PICKED', 'STARTED'):
            time.sleep(1)
            status = REDSHIFT_DATA.describe_statement(Id=response['Id'])['Status']

        while max_try > 0 and status == 'FAILED':
            max_try -= 1
            bad_sql, error = desc['QueryString'], desc['Error']
            print(f"\nDEBUG TRIAL {5 - max_try}\nBAD SQL:\n{bad_sql}\nERROR: {error}\nDEBUGGING...")
            cql = llm_debugger(bad_sql, error, params)
            q_s = cql.split('<sql>')[1].split('</sql>')[0].strip()
            print(f"\nDEBUGGED SQL:\n{q_s}")
            response = execute_query_redshift(q_s, conn_param, database)

            desc = REDSHIFT_DATA.describe_statement(Id=response['Id'])
            status = desc['Status']
            while status in ('SUBMITTED', 'PICKED', 'STARTED'):
                time.sleep(2)
                status = REDSHIFT_DATA.describe_statement(Id=response['Id'])['Status']

            if status == 'FINISHED':
                break

        if max_try == 0 and status == 'FAILED':
            print(f'DEBUGGING FAILED IN {debug_count} ATTEMPTS')
        else:
            for _ in range(5):
                try:
                    time.sleep(1)
                    stmt_result = REDSHIFT_DATA.get_statement_result(Id=response['Id'])
                    break
                except REDSHIFT_DATA.exceptions.ResourceNotFoundException:
                    time.sleep(5)

    if max_try == 0 and status == 'FAILED':
        result_csv = f'DEBUGGING FAILED IN {debug_count} ATTEMPTS. NO RESULT AVAILABLE'
    else:
        result_csv = get_redshift_table_result(stmt_result)

    return result_csv, q_s
